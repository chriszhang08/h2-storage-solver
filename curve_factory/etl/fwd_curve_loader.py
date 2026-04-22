import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable, NamedTuple

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from constants import NUM_M

from curve_factory.utils.curve_data_transformers import long_to_matrix


class CurveMatrix(NamedTuple):
    """Wide-format curve data with axis labels."""

    matrix: np.ndarray  # (n_dates, n_maturities)
    dates: np.ndarray  # 1-D array of date labels (length n_dates)
    maturities: np.ndarray  # 1-D array of maturity values (length n_maturities)


@dataclass
class FuturesDataConfig:
    """
    Configuration for loading and transforming futures curves.

    Designed for wide-format CSVs where:
      - One column is the date (e.g. 'Date')
      - All other columns are contract identifiers (absolute maturities)
    """

    date_col: str = "Date"  # Name of date column in raw CSV
    price_col_name: str = "price"  # Name to use in long-form
    contract_col_name: str = "contract"  # Name for melted contract column
    maturity_col_name: str = "maturity"  # Name for relative maturity column

    # Optional: function that takes the wide df and returns list of columns to melt
    contract_col_selector: Optional[Callable[[pd.DataFrame], List[str]]] = None

    # Optional max maturity filter (e.g. 62 months)
    max_maturity: Optional[int] = None
    spot: Optional[bool] = False  # whether this curve is a spot curve (maturity=0)

    # Optional: skipcols or drop pattern (e.g. drop year 2012 columns)
    drop_col_predicate: Optional[Callable[[str], bool]] = None

    # Optional: whether to impute missing maturities after filtering
    impute: bool = True


@dataclass
class FuturesCurveLoader:
    """
    Generic futures curve loader / transformer for wide→long format.

    Current capabilities:
      - Load CSV
      - Drop columns by predicate (e.g. year 2012)
      - Melt wide curve to long format
      - Build maturity lookup (relative maturity per date)
      - Overwrite contract labels with relative maturity index
      - Standardize columns to [date, price, maturity]
      - Optional max maturity filter

    Designed to be extended with:
      - Different date formats
      - Alternate maturity mapping logic
      - Multiple price columns, currencies, etc.
    """

    forward_config: FuturesDataConfig = field(default_factory=FuturesDataConfig)
    spot_config: Optional[FuturesDataConfig] = None

    # Internal caches (optional)
    _maturity_lut: Dict[pd.Timestamp, Dict[str, int]] = field(
        default_factory=dict, init=False
    )

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    def load_from_csv(self, path: str) -> pd.DataFrame:
        """Load raw wide-format futures data."""
        df = pd.read_csv(path)
        return df

    def preprocess_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply generic preprocessing to wide df:
          - Drop columns matching predicate (e.g. year 2012)
          - Ensure date column exists
        """
        df = df.copy()

        # Drop columns (e.g. any containing '2012')
        if self.forward_config.drop_col_predicate is not None:
            drop_cols = [
                c for c in df.columns if self.forward_config.drop_col_predicate(c)
            ]
            if drop_cols:
                df = df.drop(columns=drop_cols)

        # Basic sanity check
        if self.forward_config.date_col not in df.columns:
            raise ValueError(
                f"Date column '{self.forward_config.date_col}' not found in DataFrame."
            )

        return df

    def melt_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Melt wide-format futures df into long format:
          [date_col, contract_col_name, price_col_name]
        """
        df = df.copy()

        # Which columns to melt? Default: all except date_col
        if self.forward_config.contract_col_selector is not None:
            value_vars = self.forward_config.contract_col_selector(df)
        else:
            value_vars = [c for c in df.columns if c != self.forward_config.date_col]

        melted = df.melt(
            id_vars=[self.forward_config.date_col],
            value_vars=value_vars,
            var_name=self.forward_config.contract_col_name,
            value_name=self.forward_config.price_col_name,
        ).dropna(subset=[self.forward_config.price_col_name])

        return melted

    def _impute_missing_maturities(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure each date has a complete maturity grid 1..max_maturity by linear interpolation.

        Requires:
        - maturities are integer-like buckets (1..N)
        - within each date, price is defined for a subset of maturities
        - missing maturities should be linearly interpolated
        - outside observed maturities, hold the nearest endpoint value (np.interp behavior)
        """
        if not self.forward_config.impute:
            return df_long

        if self.forward_config.spot:
            return df_long

        max_m = int(self.forward_config.max_maturity)
        target_maturities = np.arange(1, max_m + 1, dtype=float)

        out_rows = []
        # groupby date and interpolate each curve
        for date, grp in df_long.groupby("date", sort=True):
            grp = grp.sort_values("maturity")

            orig_mats = grp["maturity"].to_numpy(dtype=float)
            orig_prices = grp["price"].to_numpy(dtype=float)

            # If only 1 point exists, flat fill across maturities
            if orig_mats.size == 1:
                imputed_prices = np.full_like(
                    target_maturities, orig_prices[0], dtype=float
                )
            else:
                # np.interp: linear interpolation; for x outside range it uses endpoint values
                pchip = PchipInterpolator(orig_mats, orig_prices)
                imputed_prices = pchip(target_maturities)

            out_rows.append(
                pd.DataFrame(
                    {
                        "date": date,
                        "maturity": target_maturities.astype(int),
                        "price": imputed_prices,
                    }
                )
            )

        df_imputed = pd.concat(out_rows, ignore_index=True)

        # Optional sanity: enforce types
        df_imputed["date"] = pd.to_datetime(df_imputed["date"])
        df_imputed["maturity"] = df_imputed["maturity"].astype(int)
        df_imputed["price"] = pd.to_numeric(df_imputed["price"], errors="coerce")

        return df_imputed.dropna(subset=["price"])

    # ------------------------------------------------------------------ #
    # MATURITY MAPPING
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_maturity_map(
        wide_df: pd.DataFrame, date_col: str
    ) -> Dict[pd.Timestamp, Dict[str, int]]:
        """
        Build mapping: {row_index (date) -> {contract_column -> maturity_index}}

        Maturity index = 1 for first non-NaN contract in a row,
                         2 for second, etc.
        """
        df = wide_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

        maturity_map: Dict[pd.Timestamp, Dict[str, int]] = {}

        for date_idx, row in df.iterrows():
            row_map: Dict[str, int] = {}
            # Count leading NaNs to align maturities
            leading_nan_count = 0
            for col in df.columns:
                if pd.isna(row[col]):
                    leading_nan_count += 1
                else:
                    break

            for col_idx, col in enumerate(df.columns, start=1):
                value = row[col]
                if pd.isna(value):
                    continue
                maturity = col_idx - leading_nan_count
                if maturity >= 1:
                    row_map[col] = maturity

            maturity_map[date_idx] = row_map

        return maturity_map

    @staticmethod
    def _overwrite_contract_with_rel_maturity(
        melted: pd.DataFrame,
        maturity_lut: Dict[pd.Timestamp, Dict[str, int]],
        date_col: str,
        contract_col: str,
        maturity_col_name: str,
        spot: bool = False,
    ) -> pd.DataFrame:
        """
        Replace absolute contract identifiers (e.g. '2015-01-02') with relative maturity index.
        """
        df = melted.copy()

        if spot:
            df[maturity_col_name] = 0
            return df

        df[date_col] = pd.to_datetime(df[date_col])

        maturities: List[Optional[int]] = []

        for _, row in df.iterrows():
            d = row[date_col]
            c = row[contract_col]

            # Lookup by date, then by contract column name
            m_for_date = maturity_lut.get(d)
            if m_for_date is None:
                maturities.append(np.nan)
            else:
                maturities.append(m_for_date.get(c, np.nan))

        df[maturity_col_name] = maturities
        return df

    # ------------------------------------------------------------------ #
    # FULL PIPELINE
    # ------------------------------------------------------------------ #

    def load_and_transform(self, path: str) -> pd.DataFrame:
        """
        Full pipeline:
          1. Load CSV
          2. Preprocess wide df (drop columns, etc.)
          3. Build maturity lookup
          4. Melt to long format
          5. Overwrite contract with relative maturity
          6. Standardize column names to [date, price, maturity]
          7. Optional max maturity filter

        Returns:
            Long-form DataFrame with columns: ['date', 'price', 'maturity']
        """
        # 1. Load
        wide_df = self.load_from_csv(path)

        # 2. Preprocess (drop 2012, etc.)
        wide_df = self.preprocess_wide(wide_df)

        # 3. Build maturity map
        maturity_lut = self._build_maturity_map(wide_df, self.forward_config.date_col)
        self._maturity_lut = maturity_lut  # cache

        # 4. Melt
        melted = self.melt_to_long(wide_df)

        # 5. Overwrite contract with relative maturity
        melted_with_maturity = self._overwrite_contract_with_rel_maturity(
            melted,
            maturity_lut,
            date_col=self.forward_config.date_col,
            contract_col=self.forward_config.contract_col_name,
            maturity_col_name=self.forward_config.maturity_col_name,
            spot=self.forward_config.spot,
        )

        # 6. Standardize column names
        df_long = melted_with_maturity.rename(
            columns={
                self.forward_config.date_col: "date",
                self.forward_config.price_col_name: "price",
                self.forward_config.maturity_col_name: "maturity",
            }
        ).drop(columns=[self.forward_config.contract_col_name])

        df_long["date"] = pd.to_datetime(df_long["date"])

        # 7. Optional maturity filter
        if self.forward_config.max_maturity is not None:
            df_long["maturity"] = pd.to_numeric(df_long["maturity"], errors="coerce")
            df_long["price"] = pd.to_numeric(df_long["price"], errors="coerce")
            df_long = df_long.dropna(subset=["maturity", "price"])

            df_long = df_long[
                df_long["maturity"] <= self.forward_config.max_maturity
            ].reset_index(drop=True)

            # Impute missing maturities for each date to create a full grid 1..max_maturity
            df_long = self._impute_missing_maturities(df_long)

        return df_long

    # ------------------------------------------------------------------ #
    # COMBINED SPOT + FORWARD PIPELINE
    # ------------------------------------------------------------------ #

    def _join_spot_and_forward(
        self, forward_df: pd.DataFrame, spot_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Concatenate spot (maturity=0) and forward long-form DataFrames, sorted by date and maturity."""
        combined = pd.concat([spot_df, forward_df], ignore_index=True)
        return combined.sort_values(["date", "maturity"]).reset_index(drop=True)

    def _filter_complete_dates(
        self, df: pd.DataFrame, include_spot: bool
    ) -> pd.DataFrame:
        """
        Keep only dates that have a full set of maturities.

        A complete date has all forward maturities 1..max_maturity, plus
        the spot (maturity=0) if include_spot is True.
        """
        expected = int(self.forward_config.max_maturity or df["maturity"].max())
        if include_spot:
            expected += 1  # account for maturity=0
        return (
            df.groupby("date")
            .filter(lambda g: g["maturity"].nunique() == expected)
            .reset_index(drop=True)
        )

    def _to_matrix(self, df: pd.DataFrame) -> CurveMatrix:
        """Pivot long-form DataFrame to a (T, M) numpy matrix sorted by date and maturity."""
        pivot = long_to_matrix(df)
        return CurveMatrix(
            matrix=pivot.to_numpy(),
            dates=pivot.index.to_numpy(),
            maturities=pivot.columns.to_numpy(),
        )

    def load_and_build_matrix(
        self,
        forward_path: str,
        spot_path: Optional[str] = None,
    ) -> CurveMatrix:
        """
        Full pipeline producing a (T, M) price matrix ready for TradingEnv.

        Steps:
          1. Load and transform forward curves  → long-form [date, price, maturity 1..M]
          2. Load and transform spot curve      → long-form [date, price, maturity=0]   (if spot_path)
          3. Join spot and forward              → combined long-form                     (if spot_path)
          4. Filter to dates with a complete maturity grid
          5. Pivot to (T, M) numpy matrix

        Args:
            forward_path: Path to forward curves CSV.
            spot_path:    Path to spot curve CSV. Requires spot_config to be set.

        Returns:
            np.ndarray of shape (T, M) where M = max_maturity (or max_maturity + 1 when spot included).

        Raises:
            ValueError: If spot_path is given but spot_config is not set.
        """
        # Step 1: Load forward curves
        forward_df = self.load_and_transform(forward_path)

        include_spot = spot_path is not None
        if include_spot:
            if self.spot_config is None:
                raise ValueError(
                    "spot_path was provided but spot_config is not set on this loader. "
                    "Pass spot_config=FuturesDataConfig(..., spot=True) when constructing FuturesCurveLoader."
                )
            # Step 2: Load spot curve (uses a loader configured with spot_config)
            spot_loader = FuturesCurveLoader(forward_config=self.spot_config)
            spot_df = spot_loader.load_and_transform(spot_path)

            # Step 3: Join
            combined_df = self._join_spot_and_forward(forward_df, spot_df)
        else:
            combined_df = forward_df

        # Step 4: Filter to complete dates
        complete_df = self._filter_complete_dates(
            combined_df, include_spot=include_spot
        )

        # Step 5: Pivot to matrix
        return self._to_matrix(complete_df)

    # ------------------------------------------------------------------ #
    # EXTENSION HOOKS
    # ------------------------------------------------------------------ #

    def get_maturity_map(self) -> Dict[pd.Timestamp, Dict[str, int]]:
        """Expose maturity lookup table."""
        return self._maturity_lut


def load_curves_etl(commodity: str) -> tuple[CurveMatrix, Dict[dt.datetime, int]]:
    """
    Load and build the forward curve matrix for a single commodity.

    Args:
        commodity: One of the keys in path_to_data (e.g. "ttf", "nlpwr").

    Returns:
        A tuple of (CurveMatrix, dates_to_idx mapping).
    """
    path_to_data = {
        "ttf": {
            "forward_path": "data/ttf/ttf_monthly_mark.csv",
            "spot_path": "data/ttf/ttf_daily_mark.csv",
        },
        "nlpwr": {
            "forward_path": "data/power/nl_monthly_mark.csv",
            "spot_path": "data/power/nl_daily_mark.csv",
        },
    }

    if commodity not in path_to_data:
        raise ValueError(
            f"Unknown commodity '{commodity}'. Choose from: {list(path_to_data)}"
        )
    paths = path_to_data[commodity]

    forward_config = FuturesDataConfig(
        date_col="Date",
        price_col_name="Price",
        contract_col_name="Contract",
        maturity_col_name="Maturity",
        max_maturity=NUM_M - 1,
        drop_col_predicate=None,
    )
    spot_config = FuturesDataConfig(
        date_col="Date",
        price_col_name="Price",
        contract_col_name="Contract",
        maturity_col_name="Maturity",
        max_maturity=1,
        spot=True,
        drop_col_predicate=None,
    )

    loader = FuturesCurveLoader(forward_config=forward_config, spot_config=spot_config)
    np_matrix: CurveMatrix = loader.load_and_build_matrix(
        forward_path=paths["forward_path"],
        spot_path=paths["spot_path"],
    )
    dates_to_idx: Dict[dt.datetime, int] = {
        date: idx for idx, date in enumerate(np_matrix.dates)
    }
    return np_matrix, dates_to_idx
