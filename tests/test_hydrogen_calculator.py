"""
Tests for SMR_Natural_Gas_Plugin and HydrogenLCOHCalculator.

Coverage
--------
- Plugin registration in sys.modules
- SMR_Natural_Gas_Plugin DCF integration (grey and blue)
- HydrogenLCOHCalculator scalar inputs
- HydrogenLCOHCalculator array inputs
- Return type consistency (scalar in → float, array in → ndarray)
- Per-instance result cache
- Economic ordering invariants
- Sensitivity to NG / power / carbon price
- DCF input dict structure (required tables and keys)
- Unit conversion helper
- Blue-Green H2 index
"""

import sys

import numpy as np
import pytest

from curve_factory.hydrogen_curve_factory import (
    HydrogenLCOHCalculator,
    _register_smr_plugin,
)
from curve_factory.hydrogen_curve_factory import SMR_Natural_Gas_Plugin
from pyH2A.Discounted_Cash_Flow import Discounted_Cash_Flow


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def calc():
    """Default HydrogenLCOHCalculator instance, shared across module tests."""
    return HydrogenLCOHCalculator()


@pytest.fixture(scope="module")
def grey_lcoh(calc):
    return calc.compute_grey_lcoh(30.0)


@pytest.fixture(scope="module")
def blue_lcoh(calc):
    return calc.compute_blue_lcoh(30.0, carbon_price=80.0)


@pytest.fixture(scope="module")
def green_lcoh(calc):
    return calc.compute_green_lcoh(50.0)


# ─── Plugin registration ─────────────────────────────────────────────────────

class TestPluginRegistration:
    def test_plugin_in_sys_modules(self):
        assert "pyH2A.Plugins.SMR_Natural_Gas_Plugin" in sys.modules

    def test_plugin_class_accessible_from_module(self):
        mod = sys.modules["pyH2A.Plugins.SMR_Natural_Gas_Plugin"]
        assert hasattr(mod, "SMR_Natural_Gas_Plugin")
        assert mod.SMR_Natural_Gas_Plugin is SMR_Natural_Gas_Plugin

    def test_register_is_idempotent(self):
        mod_before = sys.modules["pyH2A.Plugins.SMR_Natural_Gas_Plugin"]
        _register_smr_plugin()
        assert sys.modules["pyH2A.Plugins.SMR_Natural_Gas_Plugin"] is mod_before

    def test_plugin_importable_by_pyh2a(self):
        from importlib import import_module
        mod = import_module("pyH2A.Plugins.SMR_Natural_Gas_Plugin")
        cls = getattr(mod, "SMR_Natural_Gas_Plugin")
        assert cls is SMR_Natural_Gas_Plugin


# ─── DCF input dict structure ─────────────────────────────────────────────────

class TestDcfInputStructure:
    REQUIRED_SMR_TABLES = [
        "Workflow",
        "Financial Input Values",
        "Construction",
        "Technical Operating Parameters and Specifications",
        "Natural Gas SMR",
        "Direct Capital Cost - SMR",
        "Indirect Capital Cost - SMR",
        "Non-Depreciable Capital Costs",
        "Fixed Operating Costs",
        "Planned Replacement",
    ]

    REQUIRED_GREEN_TABLES = [
        "Workflow",
        "Financial Input Values",
        "Construction",
        "Technical Operating Parameters and Specifications",
        "Direct Capital Costs - PV",
        "Direct Capital Costs - Electrolyzer",
        "Non-Depreciable Capital Costs",
        "Fixed Operating Costs",
        "Utilities",
        "Electrolyzer",
        "Photovoltaic",
    ]

    def test_smr_input_contains_required_tables(self, calc):
        inp = calc._build_smr_dcf_input(30.0, blue=False)
        for table in self.REQUIRED_SMR_TABLES:
            assert table in inp, f"Missing table: {table}"

    def test_green_input_contains_required_tables(self, calc):
        inp = calc._build_green_dcf_input(50.0)
        for table in self.REQUIRED_GREEN_TABLES:
            assert table in inp, f"Missing table: {table}"

    def test_smr_ng_table_has_required_keys(self, calc):
        ng = calc._build_smr_dcf_input(30.0)["Natural Gas SMR"]
        for key in ("Spot Price ($/MWh)", "Heat Rate (MWh/kg H2)",
                    "CO2 Intensity (kg CO2/kg H2)", "CCS Capture Rate",
                    "Carbon Price ($/tonne CO2)"):
            assert key in ng

    def test_smr_ng_spot_price_propagated(self, calc):
        inp = calc._build_smr_dcf_input(42.5)
        assert inp["Natural Gas SMR"]["Spot Price ($/MWh)"]["Value"] == 42.5

    def test_blue_includes_ccs_capex(self, calc):
        grey_inp = calc._build_smr_dcf_input(30.0, blue=False)
        blue_inp = calc._build_smr_dcf_input(30.0, blue=True)
        grey_capex = grey_inp["Direct Capital Cost - SMR"]["SMR Plant Direct CAPEX ($)"]["Value"]
        blue_capex = blue_inp["Direct Capital Cost - SMR"]["SMR Plant Direct CAPEX ($)"]["Value"]
        assert blue_capex == grey_capex + calc.ccs_direct_capex_usd

    def test_blue_applies_ccs_capture_rate(self, calc):
        blue_inp = calc._build_smr_dcf_input(30.0, blue=True)
        assert blue_inp["Natural Gas SMR"]["CCS Capture Rate"]["Value"] == calc.ccs_capture_rate

    def test_grey_ccs_capture_rate_is_zero(self, calc):
        grey_inp = calc._build_smr_dcf_input(30.0, blue=False)
        assert grey_inp["Natural Gas SMR"]["CCS Capture Rate"]["Value"] == 0.0

    def test_blue_energy_penalty_in_heat_rate(self, calc):
        grey_inp = calc._build_smr_dcf_input(30.0, blue=False)
        blue_inp = calc._build_smr_dcf_input(30.0, blue=True)
        grey_hr = grey_inp["Natural Gas SMR"]["Heat Rate (MWh/kg H2)"]["Value"]
        blue_hr = blue_inp["Natural Gas SMR"]["Heat Rate (MWh/kg H2)"]["Value"]
        expected = calc.smr_ng_consumption_mwh_per_kg * (1.0 + calc.ccs_energy_penalty_frac)
        assert blue_hr == pytest.approx(expected, rel=1e-9)
        assert blue_hr > grey_hr

    def test_blue_adds_ccs_opex_table(self, calc):
        grey_inp = calc._build_smr_dcf_input(30.0, blue=False)
        blue_inp = calc._build_smr_dcf_input(30.0, blue=True)
        assert "Other Fixed Operating Cost - CCS" in blue_inp
        assert "Other Fixed Operating Cost - CCS" not in grey_inp

    def test_green_input_has_pv_capex_scaled_by_power_price(self, calc):
        inp = calc._build_green_dcf_input(50.0)
        expected_capex = calc.pv_reference_capex_per_kw * (50.0 / calc.pv_reference_lcoe_mwh)
        actual_capex = inp["Direct Capital Costs - PV"]["PV CAPEX ($/kW)"]["Value"]
        assert actual_capex == pytest.approx(expected_capex, rel=1e-9)

    def test_green_input_has_electrolyzer_plugin_in_workflow(self, calc):
        inp = calc._build_green_dcf_input(50.0)
        assert "Electrolyzer_Plugin" in inp["Workflow"]
        assert "Photovoltaic_Plugin" in inp["Workflow"]
        assert "Hourly_Irradiation_Plugin" in inp["Workflow"]

    def test_green_input_overrides_electrolyzer_efficiency(self, calc):
        inp = calc._build_green_dcf_input(50.0)
        expected_eff = 1.0 / (calc.electrolyzer_consumption_mwh_per_kg * 1000.0)
        actual_eff = inp["Electrolyzer"]["Conversion efficiency (kg H2/kWh)"]["Value"]
        assert actual_eff == pytest.approx(expected_eff, rel=1e-9)

    def test_workflow_includes_smr_plugin_for_smr(self, calc):
        wf = calc._build_smr_dcf_input(30.0)["Workflow"]
        assert "SMR_Natural_Gas_Plugin" in wf
        assert wf["SMR_Natural_Gas_Plugin"]["Type"] == "plugin"

    def test_workflow_uses_variable_opex_plugin_for_green(self, calc):
        wf = calc._build_green_dcf_input(50.0)["Workflow"]
        assert "Variable_Operating_Cost_Plugin" in wf
        assert "SMR_Natural_Gas_Plugin" not in wf

    def test_all_year_fields_equal(self, calc):
        fin = calc._build_smr_dcf_input(30.0)["Financial Input Values"]
        years = [
            fin["ref year"]["Value"],
            fin["startup year"]["Value"],
            fin["basis year"]["Value"],
            fin["current year capital costs"]["Value"],
        ]
        assert len(set(years)) == 1, "All year fields must be equal to avoid spurious inflation adjustments"

    def test_indirect_capex_has_path_to_direct(self, calc):
        indirect = calc._build_smr_dcf_input(30.0)["Indirect Capital Cost - SMR"]
        entry = list(indirect.values())[0]
        assert "Path" in entry
        assert "Direct Capital Cost - SMR" in entry["Path"]


# ─── SMR_Natural_Gas_Plugin DCF integration ──────────────────────────────────

class TestSMRPlugin:
    """Run bare DCF instances to verify the plugin inserts correct values."""

    def _run_dcf(self, calc, ng_price, carbon_price=0.0, blue=False):
        inp = calc._build_smr_dcf_input(ng_price, carbon_price, blue)
        return Discounted_Cash_Flow(inp, print_info=False, check_processing=False)

    def test_plugin_inserts_variable_operating_cost_total(self, calc):
        dcf = self._run_dcf(calc, 30.0)
        assert "Variable Operating Costs" in dcf.inp
        assert "Total" in dcf.inp["Variable Operating Costs"]

    def test_plugin_inserts_ng_feedstock_component(self, calc):
        dcf = self._run_dcf(calc, 30.0)
        assert "Natural Gas Feedstock" in dcf.inp["Variable Operating Costs"]

    def test_plugin_inserts_carbon_cost_component(self, calc):
        dcf = self._run_dcf(calc, 30.0, carbon_price=80.0)
        assert "Carbon Cost" in dcf.inp["Variable Operating Costs"]

    def test_ng_feedstock_scales_with_ng_price(self, calc):
        dcf_low  = self._run_dcf(calc, 20.0)
        dcf_high = self._run_dcf(calc, 40.0)
        low_feed  = dcf_low.inp["Variable Operating Costs"]["Natural Gas Feedstock"]["Value"]
        high_feed = dcf_high.inp["Variable Operating Costs"]["Natural Gas Feedstock"]["Value"]
        # Feedstock is linear in NG price; 40/20 = 2x
        assert np.allclose(high_feed / low_feed, 2.0, rtol=1e-6)

    def test_zero_carbon_price_gives_zero_carbon_cost(self, calc):
        dcf = self._run_dcf(calc, 30.0, carbon_price=0.0)
        carbon = dcf.inp["Variable Operating Costs"]["Carbon Cost"]["Value"]
        assert np.allclose(carbon, 0.0)

    def test_carbon_cost_scales_with_carbon_price(self, calc):
        dcf_low  = self._run_dcf(calc, 30.0, carbon_price=50.0)
        dcf_high = self._run_dcf(calc, 30.0, carbon_price=100.0)
        low_c  = dcf_low.inp["Variable Operating Costs"]["Carbon Cost"]["Value"]
        high_c = dcf_high.inp["Variable Operating Costs"]["Carbon Cost"]["Value"]
        assert np.allclose(high_c / low_c, 2.0, rtol=1e-6)

    def test_full_ccs_capture_gives_zero_carbon_cost(self, calc):
        full_capture_calc = HydrogenLCOHCalculator(ccs_capture_rate=1.0)
        dcf = self._run_dcf(full_capture_calc, 30.0, carbon_price=100.0, blue=True)
        carbon = dcf.inp["Variable Operating Costs"]["Carbon Cost"]["Value"]
        assert np.allclose(carbon, 0.0)

    def test_ng_table_entries_marked_processed(self, calc):
        dcf = self._run_dcf(calc, 30.0)
        for key, sub in dcf.inp["Natural Gas SMR"].items():
            assert "Processed" in sub, f"Entry '{key}' not marked Processed"

    def test_variable_cost_array_length_matches_inflation_factor(self, calc):
        dcf = self._run_dcf(calc, 30.0)
        total = dcf.inp["Variable Operating Costs"]["Total"]["Value"]
        assert len(total) == len(dcf.inflation_factor)

    def test_h2_cost_is_positive(self, calc):
        dcf = self._run_dcf(calc, 30.0)
        assert dcf.h2_cost > 0.0

    def test_blue_dcf_h2_cost_exceeds_grey_without_carbon_price(self, calc):
        grey_dcf = self._run_dcf(calc, 30.0, blue=False)
        blue_dcf = self._run_dcf(calc, 30.0, blue=True)
        assert blue_dcf.h2_cost > grey_dcf.h2_cost


# ─── Scalar LCOH computation ─────────────────────────────────────────────────

class TestScalarLCOH:
    def test_grey_returns_float(self, calc):
        assert isinstance(calc.compute_grey_lcoh(30.0), float)

    def test_blue_returns_float(self, calc):
        assert isinstance(calc.compute_blue_lcoh(30.0, 80.0), float)

    def test_green_returns_float(self, calc):
        assert isinstance(calc.compute_green_lcoh(50.0), float)

    def test_grey_is_positive(self, grey_lcoh):
        assert grey_lcoh > 0.0

    def test_blue_is_positive(self, blue_lcoh):
        assert blue_lcoh > 0.0

    def test_green_is_positive(self, green_lcoh):
        assert green_lcoh > 0.0

    def test_grey_plausible_range(self, grey_lcoh):
        # At 30 $/MWh NG, grey H2 should be in the $1–5/kg range.
        assert 1.0 < grey_lcoh < 5.0

    def test_blue_plausible_range(self, blue_lcoh):
        assert 1.0 < blue_lcoh < 7.0

    def test_green_plausible_range(self, green_lcoh):
        # At 50 $/MWh power, green H2 should be in the $2–10/kg range.
        assert 2.0 < green_lcoh < 10.0


# ─── Array LCOH computation ───────────────────────────────────────────────────

class TestArrayLCOH:
    NG_PRICES    = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    POWER_PRICES = np.array([30.0, 50.0, 70.0, 100.0])

    def test_grey_returns_ndarray_for_array_input(self, calc):
        result = calc.compute_grey_lcoh(self.NG_PRICES)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.NG_PRICES.shape

    def test_blue_returns_ndarray_for_array_input(self, calc):
        result = calc.compute_blue_lcoh(self.NG_PRICES, carbon_price=80.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.NG_PRICES.shape

    def test_green_returns_ndarray_for_array_input(self, calc):
        result = calc.compute_green_lcoh(self.POWER_PRICES)
        assert isinstance(result, np.ndarray)
        assert result.shape == self.POWER_PRICES.shape

    def test_grey_array_all_positive(self, calc):
        assert np.all(calc.compute_grey_lcoh(self.NG_PRICES) > 0.0)

    def test_blue_array_all_positive(self, calc):
        assert np.all(calc.compute_blue_lcoh(self.NG_PRICES, 80.0) > 0.0)

    def test_green_array_all_positive(self, calc):
        assert np.all(calc.compute_green_lcoh(self.POWER_PRICES) > 0.0)

    def test_grey_2d_array_preserves_shape(self, calc):
        prices_2d = np.array([[10.0, 20.0], [30.0, 40.0]])
        result = calc.compute_grey_lcoh(prices_2d)
        assert result.shape == prices_2d.shape

    def test_array_results_consistent_with_scalar(self, calc):
        array_result = calc.compute_grey_lcoh(self.NG_PRICES)
        for i, price in enumerate(self.NG_PRICES):
            scalar_result = calc.compute_grey_lcoh(float(price))
            assert array_result[i] == pytest.approx(scalar_result, rel=1e-9)


# ─── Economic ordering invariants ────────────────────────────────────────────

class TestEconomicInvariants:
    """Monotonicity and comparative-statics checks grounded in energy economics."""

    def test_grey_lcoh_increases_with_ng_price(self, calc):
        prices = np.linspace(10.0, 80.0, 8)
        lcoh = calc.compute_grey_lcoh(prices)
        assert np.all(np.diff(lcoh) > 0), "Grey LCOH must be strictly increasing in NG price"

    def test_blue_lcoh_increases_with_ng_price(self, calc):
        prices = np.linspace(10.0, 80.0, 8)
        lcoh = calc.compute_blue_lcoh(prices, carbon_price=0.0)
        assert np.all(np.diff(lcoh) > 0)

    def test_green_lcoh_increases_with_power_price(self, calc):
        prices = np.linspace(20.0, 100.0, 8)
        lcoh = calc.compute_green_lcoh(prices)
        assert np.all(np.diff(lcoh) > 0)

    def test_blue_lcoh_increases_slower_than_grey_with_carbon_price(self, calc):
        # The carbon tax is applied to uncaptured CO2 only.  At 90 % CCS capture,
        # blue H2 bears ~0.9 kg CO2/kg H2 while grey bears ~9.0 kg CO2/kg H2.
        # Both LCOHs rise with carbon price, but grey rises ~10x faster.
        carbon_prices = np.linspace(0.0, 200.0, 8)
        grey_with_tax = np.array([
            calc._run_dcf(calc._build_smr_dcf_input(30.0, carbon_price=c, blue=False))
            for c in carbon_prices
        ])
        blue_lcoh = np.array([calc.compute_blue_lcoh(30.0, c) for c in carbon_prices])
        grey_slope = np.diff(grey_with_tax).mean()
        blue_slope = np.diff(blue_lcoh).mean()
        assert grey_slope > blue_slope, (
            f"Grey slope ({grey_slope:.4f}) should exceed blue slope ({blue_slope:.4f})"
        )

    def test_blue_exceeds_grey_with_zero_carbon_price(self, calc):
        # Without a carbon price, CCS costs money without offsetting benefit.
        grey = calc.compute_grey_lcoh(30.0)
        blue = calc.compute_blue_lcoh(30.0, carbon_price=0.0)
        assert blue > grey

    def test_high_carbon_price_makes_grey_more_expensive_than_blue(self, calc):
        # When the same carbon tax is applied to both pathways, the full 9 kg CO2/kg H2
        # on grey makes it costlier than blue (which only bears ~0.9 kg CO2/kg H2 after
        # 90 % CCS capture).  Crossover occurs around $150–200/tCO2.
        grey_with_tax = calc._run_dcf(
            calc._build_smr_dcf_input(30.0, carbon_price=300.0, blue=False)
        )
        blue_with_tax = calc.compute_blue_lcoh(30.0, carbon_price=300.0)
        assert grey_with_tax > blue_with_tax, (
            "At 300 $/tCO2 grey H2 (full carbon tax) should exceed blue H2 LCOH"
        )

    def test_grey_lcoh_linear_in_ng_price(self, calc):
        # The feedstock term is linear; LCOH = fixed_component + slope * ng_price.
        prices = np.array([20.0, 40.0, 60.0])
        lcoh = calc.compute_grey_lcoh(prices)
        # Check linearity: midpoint value should equal average of endpoints.
        assert lcoh[1] == pytest.approx((lcoh[0] + lcoh[2]) / 2.0, rel=1e-4)

    def test_blue_lcoh_linear_in_ng_price(self, calc):
        prices = np.array([20.0, 40.0, 60.0])
        lcoh = calc.compute_blue_lcoh(prices, carbon_price=0.0)
        assert lcoh[1] == pytest.approx((lcoh[0] + lcoh[2]) / 2.0, rel=1e-4)

    def test_green_lcoh_linear_in_power_price(self, calc):
        prices = np.array([30.0, 60.0, 90.0])
        lcoh = calc.compute_green_lcoh(prices)
        assert lcoh[1] == pytest.approx((lcoh[0] + lcoh[2]) / 2.0, rel=1e-4)

    def test_grey_capex_contributes_positively(self, calc):
        # A plant with lower CAPEX should yield a lower LCOH.
        low_capex = HydrogenLCOHCalculator(smr_direct_capex_usd=100_000_000.0)
        high_capex = HydrogenLCOHCalculator(smr_direct_capex_usd=300_000_000.0)
        assert low_capex.compute_grey_lcoh(30.0) < high_capex.compute_grey_lcoh(30.0)

    def test_higher_ccs_capture_rate_reduces_blue_at_high_carbon_price(self, calc):
        low_capture  = HydrogenLCOHCalculator(ccs_capture_rate=0.56)
        high_capture = HydrogenLCOHCalculator(ccs_capture_rate=0.95)
        assert (
            high_capture.compute_blue_lcoh(30.0, carbon_price=200.0)
            < low_capture.compute_blue_lcoh(30.0, carbon_price=200.0)
        )


# ─── Caching ─────────────────────────────────────────────────────────────────

class TestCaching:
    def test_grey_cache_populated_after_call(self, calc):
        fresh = HydrogenLCOHCalculator()
        assert len(fresh._smr_cache) == 0
        fresh.compute_grey_lcoh(30.0)
        assert len(fresh._smr_cache) == 1

    def test_blue_cache_populated_after_call(self, calc):
        fresh = HydrogenLCOHCalculator()
        fresh.compute_blue_lcoh(30.0, carbon_price=80.0)
        assert len(fresh._smr_cache) == 1

    def test_green_cache_populated_after_call(self):
        fresh = HydrogenLCOHCalculator()
        assert len(fresh._green_cache) == 0
        fresh.compute_green_lcoh(50.0)
        assert len(fresh._green_cache) == 1

    def test_repeated_scalar_call_returns_identical_result(self, calc):
        a = calc.compute_grey_lcoh(25.0)
        b = calc.compute_grey_lcoh(25.0)
        assert a == b

    def test_cache_is_per_instance(self):
        c1 = HydrogenLCOHCalculator()
        c2 = HydrogenLCOHCalculator()
        c1.compute_grey_lcoh(30.0)
        assert len(c2._smr_cache) == 0

    def test_grey_and_blue_share_smr_cache_with_distinct_keys(self):
        fresh = HydrogenLCOHCalculator()
        fresh.compute_grey_lcoh(30.0)
        fresh.compute_blue_lcoh(30.0, carbon_price=80.0)
        assert len(fresh._smr_cache) == 2

    def test_different_rounded_prices_produce_separate_entries(self):
        fresh = HydrogenLCOHCalculator()
        fresh.compute_grey_lcoh(30.0)
        fresh.compute_grey_lcoh(31.0)
        assert len(fresh._smr_cache) == 2

    def test_prices_within_rounding_share_cache_entry(self):
        fresh = HydrogenLCOHCalculator()
        # 30.001 and 30.004 both round to 30.00 at 2 d.p.
        r1 = fresh.compute_grey_lcoh(30.001)
        r2 = fresh.compute_grey_lcoh(30.004)
        assert r1 == r2
        assert len(fresh._smr_cache) == 1


# ─── Blue-Green H2 index ─────────────────────────────────────────────────────

class TestBlueGreenIndex:
    def test_returns_ndarray_for_array_inputs(self, calc):
        ng = np.array([20.0, 30.0, 40.0])
        result = calc.compute_blue_green_h2_index(ng, carbon_price=80.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == ng.shape

    def test_index_is_positive(self, calc):
        ng = np.array([20.0, 30.0, 40.0])
        result = calc.compute_blue_green_h2_index(ng)
        assert np.all(result > 0)

    def test_index_uses_weighted_contributions(self, calc):
        from constants import PV_LCOE, BLUE_H2_CONTRIB, GREEN_H2_CONTRIB
        ng = np.array([20.0, 30.0, 40.0])
        co2 = 80.0
        blue = calc.compute_blue_lcoh(ng, co2)
        green = calc.compute_green_lcoh(PV_LCOE, co2)
        expected = BLUE_H2_CONTRIB * blue + GREEN_H2_CONTRIB * green
        result = calc.compute_blue_green_h2_index(ng, carbon_price=co2)
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_seasonality_compression(self, calc):
        ng = np.array([20.0, 30.0, 40.0])
        full = calc.compute_blue_green_h2_index(ng, seasonality_alpha=1.0)
        compressed = calc.compute_blue_green_h2_index(ng, seasonality_alpha=0.5)
        # At alpha=0.5, variance around mean is halved.
        assert np.std(compressed) == pytest.approx(np.std(full) * 0.5, rel=1e-9)


# ─── Unit conversion ─────────────────────────────────────────────────────────

class TestUnitConversion:
    def test_lcoh_per_kg_to_per_mwh_scalar(self):
        # 1 $/kg H2 × (1 kg / 0.03333 MWh) = ~30 $/MWh
        result = HydrogenLCOHCalculator.lcoh_per_kg_to_per_mwh(1.0)
        assert result == pytest.approx(1.0 / (33.33 / 1000.0), rel=1e-4)

    def test_lcoh_per_kg_to_per_mwh_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = HydrogenLCOHCalculator.lcoh_per_kg_to_per_mwh(arr)
        assert result.shape == arr.shape
        assert result[1] == pytest.approx(result[0] * 2.0, rel=1e-9)

    def test_hhv_basis_gives_lower_mwh_price(self):
        lcoh = 2.0
        lhv = HydrogenLCOHCalculator.lcoh_per_kg_to_per_mwh(lcoh, heating_value_kwh_per_kg=33.33)
        hhv = HydrogenLCOHCalculator.lcoh_per_kg_to_per_mwh(lcoh, heating_value_kwh_per_kg=39.41)
        # HHV > LHV so $/MWh on HHV basis is lower than LHV basis
        assert hhv < lhv

    def test_conversion_is_static_method(self):
        # Should be callable without an instance
        result = HydrogenLCOHCalculator.lcoh_per_kg_to_per_mwh(2.0)
        assert result > 0.0


# ─── Scenario overrides ───────────────────────────────────────────────────────

class TestScenarioOverrides:
    """Verify that dataclass field overrides propagate correctly into DCF runs."""

    def test_higher_irr_increases_lcoh(self):
        low_irr  = HydrogenLCOHCalculator(irr=0.05)
        high_irr = HydrogenLCOHCalculator(irr=0.15)
        assert high_irr.compute_grey_lcoh(30.0) > low_irr.compute_grey_lcoh(30.0)

    def test_larger_plant_reduces_per_kg_capex_contribution(self):
        # Doubling capacity while keeping absolute CAPEX constant halves the
        # per-kg capital burden (more kg produced per dollar of plant).
        base = HydrogenLCOHCalculator(
            smr_plant_capacity_kg_per_day=50_000.0,
            smr_direct_capex_usd=200_000_000.0,
        )
        double = HydrogenLCOHCalculator(
            smr_plant_capacity_kg_per_day=100_000.0,
            smr_direct_capex_usd=200_000_000.0,
        )
        assert double.compute_grey_lcoh(30.0) < base.compute_grey_lcoh(30.0)

    def test_ccs_opex_zero_reduces_blue_lcoh(self):
        with_opex    = HydrogenLCOHCalculator(ccs_opex_per_kg=0.47)
        without_opex = HydrogenLCOHCalculator(ccs_opex_per_kg=0.0)
        assert without_opex.compute_blue_lcoh(30.0, 0.0) < with_opex.compute_blue_lcoh(30.0, 0.0)

    def test_ng_consumption_rate_scales_feedstock_cost(self):
        low_cons  = HydrogenLCOHCalculator(smr_ng_consumption_mwh_per_kg=0.040)
        high_cons = HydrogenLCOHCalculator(smr_ng_consumption_mwh_per_kg=0.056)
        # The ratio of LCOH differences at two prices should reflect the
        # consumption difference, since fixed costs cancel out.
        delta_low  = low_cons.compute_grey_lcoh(40.0)  - low_cons.compute_grey_lcoh(20.0)
        delta_high = high_cons.compute_grey_lcoh(40.0) - high_cons.compute_grey_lcoh(20.0)
        assert delta_high / delta_low == pytest.approx(0.056 / 0.040, rel=1e-4)

    def test_pem_consumption_rate_scales_green_cost(self):
        low_cons  = HydrogenLCOHCalculator(electrolyzer_consumption_mwh_per_kg=0.050)
        high_cons = HydrogenLCOHCalculator(electrolyzer_consumption_mwh_per_kg=0.060)
        delta_low  = low_cons.compute_green_lcoh(60.0)  - low_cons.compute_green_lcoh(30.0)
        delta_high = high_cons.compute_green_lcoh(60.0) - high_cons.compute_green_lcoh(30.0)
        assert delta_high / delta_low == pytest.approx(0.060 / 0.050, rel=1e-4)


# ─── Summary method ──────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_returns_string(self, calc):
        assert isinstance(calc.summary(), str)

    def test_summary_contains_pathway_labels(self, calc):
        s = calc.summary()
        assert "Grey" in s
        assert "Blue" in s
        assert "Green" in s

    def test_summary_contains_price_inputs(self, calc):
        s = calc.summary(ng_price=25.0, power_price=45.0, carbon_price=60.0)
        assert "25.0" in s
        assert "45.0" in s
        assert "60.0" in s
