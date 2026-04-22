"""
HydrogenLCOHCalculator
======================
Discounted cash-flow LCOH calculator that converts forward price curves
(natural gas, electricity, carbon) into Levelized Cost of Hydrogen (LCOH)
in $/kg for three production pathways:

    Grey  — unabated Steam Methane Reforming (SMR)
    Blue  — SMR with Carbon Capture and Storage (CCS)
    Green — PEM water electrolysis powered by grid/renewable electricity

Backed by the pyH2A discounted cash-flow engine (DOE H2A methodology).
All physical and economic constants carry inline literature citations.
Input and output arrays share the same shape, so a (T, M) forward-curve
matrix produces a (T, M) LCOH matrix and a (T,) spot vector produces a
(T,) LCOH vector.

Hydrogen demand is deliberately excluded from this class.  Demand is
discrete and non-deterministic (driven by offtake contracts, policy
mandates, and willingness-to-pay of downstream sectors) and is modelled
separately in ``demand_center.py``.

Plugin architecture
-------------------
``SMR_Natural_Gas_Plugin`` is a pyH2A workflow plugin defined in this
module and injected into ``sys.modules`` at import time so pyH2A can
discover it by name.  It replaces the standard
``Variable_Operating_Cost_Plugin`` for SMR pathways and accepts the
natural gas spot price as a first-class input rather than a static
lookup table.

Key references
--------------
[1] Argonne National Laboratory, "Updates of Hydrogen Production from
    SMR Process", GREET Model, 2019.
    https://greet.anl.gov/files/smr_h2_2019
[2] IEA, "Global Hydrogen Review 2023", Sep 2023.
    https://www.iea.org/reports/global-hydrogen-review-2023
[3] IEA, "Global Hydrogen Review 2023 — Assumptions Annex", 2023.
    https://iea.blob.core.windows.net/assets/2ceb17b8-474f-4154-aab5-4d898f735c17/IEAGHRassumptions_final.pdf
[4] DOE Hydrogen Program Record 24005, "Clean Hydrogen Production Cost —
    PEM Electrolyzer", May 2024.
    https://www.hydrogen.energy.gov/docs/hydrogenprogramlibraries/pdfs/24005-clean-hydrogen-production-cost-pem-electrolyzer.pdf
[5] IEAGHG, "Techno-Economic Evaluation of SMR Based Standalone
    (Merchant) Hydrogen Plant with CCS", 2017.
    https://ieaghg.org/publications/techno-economic-evaluation-of-smr-based-standalone-merchant-hydrogen-plant-with-ccs/
[6] UK BEIS, "Hydrogen Production Costs 2021", Aug 2021.
    https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1011506/Hydrogen_Production_Costs_2021.pdf
[7] IEEFA, "Reality Check on CO2 Emissions Capture at Hydrogen-From-Gas
    Plants", Feb 2022.
    https://ieefa.org/wp-content/uploads/2022/02/Reality-Check-on-CO2-Emissions-Capture-at-Hydrogen-From-Gas-Plants_February-2022.pdf
"""

import copy
import os
import sys
import types
from dataclasses import dataclass

import numpy as np

from pyH2A.Discounted_Cash_Flow import Discounted_Cash_Flow
from pyH2A.Utilities.input_modification import convert_input_to_dictionary, insert

_BLUE_SMR_CONFIG = 'smr_blue_h2_params.md'
_GREEN_SMR_CONFIG = 'pv_e_green_h2_params.md'

from constants import PV_LCOE, GREEN_H2_CONTRIB, BLUE_H2_CONTRIB


# ─── pyH2A Plugin: SMR with dynamic natural gas pricing ──────────────────────

class SMR_Natural_Gas_Plugin:
    """
    pyH2A workflow plugin for Steam Methane Reforming with a dynamic NG price.

    Replaces ``Variable_Operating_Cost_Plugin`` for SMR pathways.  Reads the
    natural gas spot price and plant parameters from the ``Natural Gas SMR``
    config table, then inserts an inflation-adjusted annual feedstock cost
    array into ``Variable Operating Costs > Total > Value``.

    An optional carbon cost is added for uncaptured CO₂ when a carbon price
    and/or partial CCS capture rate is specified.

    Consumed from dcf.inp
    ---------------------
    Natural Gas SMR > Spot Price ($/MWh) > Value : float
        Natural gas spot price in $/MWh (same currency unit as other curves).
    Natural Gas SMR > Heat Rate (MWh/kg H2) > Value : float
        Total NG consumed per kg H₂ produced (feedstock + fuel).
        For blue hydrogen, include the CCS energy penalty in this value.
    Natural Gas SMR > CO2 Intensity (kg CO2/kg H2) > Value : float
        Unabated CO₂ emissions per kg H₂ from SMR combustion.
    Natural Gas SMR > CCS Capture Rate > Value : float
        Fraction of CO₂ permanently captured (0.0 = grey, 0.9 = blue).
    Natural Gas SMR > Carbon Price ($/tonne CO2) > Value : float
        Carbon price applied to uncaptured emissions.

    Inserted into dcf.inp
    ---------------------
    Variable Operating Costs > Total > Value : ndarray
        Flat cost array (length = n_years); DCF multiplies by inflation_factor.
    Variable Operating Costs > Natural Gas Feedstock > Value : ndarray
        NG feedstock component only.
    Variable Operating Costs > Carbon Cost > Value : ndarray
        Carbon emission cost component.
    """

    def __init__(self, dcf, print_info):
        cfg          = dcf.inp['Natural Gas SMR']
        ng_price     = cfg['Spot Price ($/MWh)']['Value']
        heat_rate    = cfg['Heat Rate (MWh/kg H2)']['Value']
        co2_int      = cfg['CO2 Intensity (kg CO2/kg H2)']['Value']
        ccs_rate     = cfg['CCS Capture Rate']['Value']
        carbon_price = cfg['Carbon Price ($/tonne CO2)']['Value']

        # Output per year inserted by Production_Scaling_Plugin (Position 1).
        output_per_year = (
            dcf.inp['Technical Operating Parameters and Specifications']
                   ['Output per Year']['Value']
        )

        # --- Natural gas feedstock cost ---
        # Insert a FLAT array scaled to the ref-year via inflation_correction.
        # The DCF core function variable_operating_costs() applies the annual
        # inflation_factor, so we must NOT pre-multiply by it here.
        base_ng   = ng_price * heat_rate * output_per_year * dcf.inflation_correction
        ng_array  = np.ones(len(dcf.inflation_factor)) * base_ng

        # --- Carbon cost on uncaptured CO₂ ---
        # uncaptured_co2 in kg CO₂ / kg H₂; divide by 1000 to convert to tonnes.
        uncaptured = co2_int * (1.0 - ccs_rate)
        base_carbon  = (uncaptured / 1000.0) * carbon_price * output_per_year * dcf.inflation_correction
        carbon_array = np.ones(len(dcf.inflation_factor)) * base_carbon

        total_var = ng_array + carbon_array

        insert(dcf, 'Variable Operating Costs', 'Total',
               'Value', total_var, __name__, print_info=print_info)
        insert(dcf, 'Variable Operating Costs', 'Natural Gas Feedstock',
               'Value', ng_array, __name__, print_info=print_info)
        insert(dcf, 'Variable Operating Costs', 'Carbon Cost',
               'Value', carbon_array, __name__, print_info=print_info)

        # Mark every entry in the custom table as processed so the DCF
        # check_processing pass does not emit spurious warnings.
        for key in cfg:
            cfg[key]['Processed'] = 'Yes'


def _register_smr_plugin() -> None:
    """Inject SMR_Natural_Gas_Plugin into sys.modules for pyH2A discovery.

    pyH2A resolves plugins via ``import_module('pyH2A.Plugins.<name>')``.
    We create a lightweight module stub so the import succeeds without
    placing any files inside the installed pyH2A package.
    """
    module_name = 'pyH2A.Plugins.SMR_Natural_Gas_Plugin'
    if module_name not in sys.modules:
        mod = types.ModuleType(module_name)
        mod.SMR_Natural_Gas_Plugin = SMR_Natural_Gas_Plugin  # type: ignore[attr-defined]
        sys.modules[module_name] = mod


# Register at import time so pyH2A can find the plugin.
_register_smr_plugin()


# ─── LCOH Calculator ─────────────────────────────────────────────────────────

@dataclass
class HydrogenLCOHCalculator:
    """
    Discounted cash-flow LCOH calculator for grey, blue, and green hydrogen.

    Each pathway is backed by a full pyH2A DCF run (DOE H2A methodology),
    including MACRS depreciation, equity/debt financing, federal and state
    taxes, salvage, and working capital.  Results are cached by rounded
    input prices so repeated calls with the same price incur no overhead.

    Override at construction time for scenario / sensitivity analysis::

        conservative = HydrogenLCOHCalculator(ccs_capture_rate=0.56)
        optimistic   = HydrogenLCOHCalculator(ccs_capture_rate=0.95)

    All prices follow the convention of the input curves:
        - gas / power prices in $/MWh  (or €/MWh — currency-neutral)
        - carbon prices in $/tCO2      (or €/tCO2)
        - LCOH output in $/kg H2       (same currency as inputs)

    Plant-scale parameters
    ----------------------
    SMR and PEM plants are sized at 50 tpd H₂ by default, consistent with
    the DOE H2A central production case.  Absolute CAPEX figures are taken
    from IEA GHR 2023 and DOE H2A v3.2.  Override these to model different
    plant scales or technology vintages.
    """

    # =================================================================== #
    #  SMR — Steam Methane Reforming (Grey / Blue shared constants)       #
    # =================================================================== #

    # Total natural gas consumed per kg H2 produced (feedstock + fuel).
    # Literature range: 0.045–0.053 MWh/kg; ~45–53 kWh/kg.
    # Midpoint 0.048 MWh/kg (≈ 173 MJ/kg) used as default.
    # Sources: Argonne GREET 2019 [1]; IEA GHR 2023 [2]
    smr_ng_consumption_mwh_per_kg: float = 0.048

    # Unabated CO2 emitted per kg H2 from SMR (no capture).
    # Literature range: 8–10 kg CO2/kg H2; central estimate 9.0.
    # Sources: IEEFA 2022 [7]; IEA GHR 2023 [2]
    smr_co2_intensity_kg_per_kg: float = 9.0

    # SMR plant design capacity (kg H2 / day).
    # 50 tpd corresponds to the DOE H2A central SMR reference plant.
    smr_plant_capacity_kg_per_day: float = 50_000.0

    # Operating capacity factor for SMR (fraction of design capacity).
    smr_capacity_factor: float = 0.90

    # Total direct CAPEX for the SMR plant including reformer, WGS reactor,
    # PSA unit, and balance-of-plant, in 2024 USD.
    # Source: DOE H2A v3.2; IEA GHR 2023 Assumptions Annex [3]
    smr_direct_capex_usd: float = 200_000_000.0

    # Indirect CAPEX as a fraction of direct (engineering, contingency, etc.).
    smr_indirect_capex_frac: float = 0.33

    # Fixed O&M: operating staff headcount and burdened hourly rate.
    smr_staff: int = 15
    smr_hourly_labor_usd: float = 50.0

    # Land requirements for the SMR facility.
    smr_land_acres: float = 5.0
    smr_land_cost_per_acre: float = 10_000.0

    # =================================================================== #
    #  CCS — Carbon Capture & Storage (Grey → Blue add-on)               #
    # =================================================================== #

    # Fraction of total CO2 captured and permanently stored.
    # Literature range: 56 % (syngas-only) to 96 % (syngas + flue-gas).
    # Default 90 % represents a modern post-combustion + syngas scheme.
    # Sources: IEAGHG 2017 [5]; IEA GHR 2023 [2]
    ccs_capture_rate: float = 0.90

    # Additional annual OPEX for CO2 capture, compression, transport, storage,
    # expressed as $/kg H2.
    # Sources: IEA GHR 2023 [2]; Thunder Said Energy cost model
    ccs_opex_per_kg: float = 0.47

    # Energy penalty from CCS as a fraction of the base SMR gas consumption.
    # The capture plant's regeneration heat increases total gas use by ~5–14 %.
    # Sources: ScienceDirect comparative analysis (H₂ production + CCS)
    ccs_energy_penalty_frac: float = 0.10

    # Additional direct CAPEX for CCS equipment (absorbers, compressors,
    # CO2 storage infrastructure), in 2024 USD.
    # Source: IEAGHG 2017 [5]
    ccs_direct_capex_usd: float = 80_000_000.0

    # =================================================================== #
    #  PEM Electrolysis (Green)                                           #
    # =================================================================== #

    # System electricity consumption per kg H2 (stack + balance-of-plant).
    # Stack ~51 kWh/kg + BoP ~4.2 kWh/kg ≈ 55 kWh/kg → 0.055 MWh/kg.
    # Sources: DOE Program Record 24005, May 2024 [4]
    electrolyzer_consumption_mwh_per_kg: float = 0.055

    # PEM plant design capacity (kg H2 / day), matched to SMR plant for
    # a like-for-like comparison.
    pem_plant_capacity_kg_per_day: float = 50_000.0

    # Operating capacity factor for PEM (high, assumes continuous grid supply).
    pem_capacity_factor: float = 0.97

    # Total direct CAPEX for the PEM plant including stack, BoP, and power
    # electronics, in 2024 USD.
    # Basis: ~$1,700/kW × ~114 MW nominal power for 50 tpd at 55 kWh/kg.
    # Sources: IEA GHR 2023 Assumptions Annex [3]; DOE Record 24005 [4]
    pem_direct_capex_usd: float = 500_000_000.0

    # Indirect CAPEX fraction for PEM (lower than SMR — less civil works).
    pem_indirect_capex_frac: float = 0.20

    # Fixed O&M: PEM requires fewer staff than SMR.
    pem_staff: int = 10
    pem_hourly_labor_usd: float = 50.0

    # Land requirements for the PEM facility.
    pem_land_acres: float = 10.0
    pem_land_cost_per_acre: float = 10_000.0

    # =================================================================== #
    #  PV+E reference calibration (Green pathway via Photovoltaic_Plugin) #
    # =================================================================== #

    # The green pathway uses pyH2A's PV+Electrolyzer workflow (Barstow, CA
    # TMY solar resource).  ``power_price`` is mapped to PV CAPEX via LCOE
    # equivalence: a PV system at ``pv_reference_capex_per_kw`` produces
    # electricity at an effective LCOE of ``pv_reference_lcoe_mwh`` when
    # run through the full DCF.  The scaling is linear:
    #
    #     PV_CAPEX($/kW) = reference_capex × (power_price / reference_lcoe)
    #
    # Calibrated from the 210613_PV_E.md example (Barstow, CA; 20% CF).
    pv_reference_capex_per_kw: float = 1200.0   # $/kW at reference LCOE
    pv_reference_lcoe_mwh: float = 66.4         # $/MWh, effective PV LCOE at reference CAPEX

    # =================================================================== #
    #  Shared financial parameters                                        #
    # =================================================================== #

    # Plant economic life in years.
    plant_life_years: int = 20

    # After-tax real internal rate of return (equity hurdle rate).
    irr: float = 0.08

    # Fraction of project financed with equity (remainder is debt).
    equity_fraction: float = 0.40

    # Annual inflation rate applied to operating costs over plant life.
    inflation_rate: float = 0.019

    # Corporate tax rates.
    federal_tax: float = 0.21
    state_tax: float = 0.06

    # =================================================================== #
    #  Internal state (not part of the public parameter schema)           #
    # =================================================================== #

    def __post_init__(self):
        # Per-instance result caches keyed by rounded input prices.
        # Shared between grey and blue runs (key includes a blue flag).
        self._smr_cache: dict = {}
        self._green_cache: dict = {}

    # =================================================================== #
    #  pyH2A DCF input builders                                          #
    # =================================================================== #

    def _financial_input_values(self) -> dict:
        """Build the Financial Input Values table shared by all DCF runs.

        All year fields are set to the same value (2024) so that pyH2A's
        CEPCI, GDP deflator, and labor price index adjustors all evaluate
        to 1.0.  This ensures that cost inputs are used at face value
        without historical inflation re-basing.
        """
        return {
            'ref year':                    {'Full Name': 'Reference year',                           'Value': 2024},
            'startup year':                {'Full Name': 'Assumed start-up year',                    'Value': 2024},
            'basis year':                  {'Full Name': 'Basis year',                               'Value': 2024},
            'current year capital costs':  {'Full Name': 'Current year for capital costs',           'Value': 2024},
            'startup time':                {'Full Name': 'Start-up Time (years)',                    'Value': 1},
            'plant life':                  {'Full Name': 'Plant life (years)',                       'Value': self.plant_life_years},
            'depreciation length':         {'Full Name': 'Depreciation Schedule Length (years)',     'Value': 20},
            'depreciation type':           {'Full Name': 'Depreciation Type',                       'Value': 'MACRS'},
            'equity':                      {'Full Name': '% Equity Financing',                      'Value': self.equity_fraction},
            'interest':                    {'Full Name': 'Interest rate on debt (%)',                'Value': 0.037},
            'debt':                        {'Full Name': 'Debt period',                              'Value': 'Constant'},
            'startup cost fixed':          {'Full Name': '% of Fixed Costs During Start-up',        'Value': 1.0},
            'startup revenues':            {'Full Name': '% of Revenues During Start-up',           'Value': 0.75},
            'startup cost variable':       {'Full Name': '% of Variable Costs During Start-up',     'Value': 0.75},
            'decommissioning':             {'Full Name': 'Decommissioning costs (% dep. capital)',   'Value': 0.10},
            'salvage':                     {'Full Name': 'Salvage value (% total capital)',          'Value': 0.10},
            'inflation':                   {'Full Name': 'Inflation rate (%)',                       'Value': self.inflation_rate},
            'irr':                         {'Full Name': 'After-tax Real IRR (%)',                   'Value': self.irr},
            'state tax':                   {'Full Name': 'State Taxes (%)',                          'Value': self.state_tax},
            'federal tax':                 {'Full Name': 'Federal Taxes (%)',                        'Value': self.federal_tax},
            'working capital':             {'Full Name': 'Working Capital (% yearly op. cost change)', 'Value': 0.15},
        }

    def _build_smr_dcf_input(
        self,
        ng_price_mwh: float,
        carbon_price: float = 0.0,
        blue: bool = False,
    ) -> dict:
        """Dispatch to the grey or blue SMR input builder.

        Kept as a single entry-point for backward compatibility.  Delegates
        to ``_build_grey_dcf_input`` or ``_build_blue_dcf_input`` based on
        the *blue* flag.
        """
        if blue:
            return self._build_blue_dcf_input(ng_price_mwh, carbon_price)
        return self._build_grey_dcf_input(ng_price_mwh, carbon_price)

    def _build_grey_dcf_input(
        self,
        ng_price_mwh: float,
        carbon_price: float = 0.0,
    ) -> dict:
        """Build a complete pyH2A input dictionary for a grey SMR LCOH run.

        All tables are constructed from dataclass fields so that grey
        hydrogen requires no external config file.

        Parameters
        ----------
        ng_price_mwh : float
            Natural gas spot price in $/MWh.
        carbon_price : float
            Carbon price in $/tonne CO2.  Applied to uncaptured emissions.
        """
        return {
            'Workflow': {
                'Production_Scaling_Plugin':           {'Type': 'plugin',   'Position': 1},
                'production_scaling':                  {'Type': 'function', 'Position': 2},
                'Capital_Cost_Plugin':                 {'Type': 'plugin',   'Position': 3},
                'initial_equity_depreciable_capital':  {'Type': 'function', 'Position': 4},
                'non_depreciable_capital_costs':       {'Type': 'function', 'Position': 5},
                'Replacement_Plugin':                  {'Type': 'plugin',   'Position': 6},
                'replacement_costs':                   {'Type': 'function', 'Position': 7},
                'Fixed_Operating_Cost_Plugin':         {'Type': 'plugin',   'Position': 8},
                'fixed_operating_costs':               {'Type': 'function', 'Position': 9},
                'SMR_Natural_Gas_Plugin':              {'Type': 'plugin',   'Position': 10},
                'variable_operating_costs':            {'Type': 'function', 'Position': 11},
            },
            'Financial Input Values': self._financial_input_values(),
            'Construction': {
                'capital perc 1st': {'Full Name': '% of Capital Spent in Year 1', 'Value': 1.0},
            },
            'Technical Operating Parameters and Specifications': {
                'Plant Design Capacity (kg of H2/day)': {'Value': self.smr_plant_capacity_kg_per_day},
                'Operating Capacity Factor (%)':        {'Value': self.smr_capacity_factor},
            },
            'Natural Gas SMR': {
                'Spot Price ($/MWh)':            {'Value': ng_price_mwh},
                'Heat Rate (MWh/kg H2)':         {'Value': self.smr_ng_consumption_mwh_per_kg},
                'CO2 Intensity (kg CO2/kg H2)':  {'Value': self.smr_co2_intensity_kg_per_kg},
                'CCS Capture Rate':              {'Value': 0.0},
                'Carbon Price ($/tonne CO2)':    {'Value': carbon_price},
            },
            'Direct Capital Cost - SMR': {
                'SMR Plant Direct CAPEX ($)': {'Value': self.smr_direct_capex_usd},
            },
            'Indirect Capital Cost - SMR': {
                'SMR Indirect CAPEX (fraction of direct)': {
                    'Value': self.smr_indirect_capex_frac,
                    'Path':  'Direct Capital Cost - SMR > SMR Plant Direct CAPEX ($) > Value',
                },
            },
            'Non-Depreciable Capital Costs': {
                'Cost of land ($ per acre)': {'Value': self.smr_land_cost_per_acre},
                'Land required (acres)':     {'Value': self.smr_land_acres},
            },
            'Fixed Operating Costs': {
                'staff':            {'Full Name': 'Operating staff headcount', 'Value': self.smr_staff},
                'hourly labor cost': {'Full Name': 'Burdened labor cost ($/hr)', 'Value': self.smr_hourly_labor_usd},
            },
            'Planned Replacement': {},
        }

    def _build_blue_dcf_input(
        self,
        ng_price_mwh: float,
        carbon_price: float = 0.0,
    ) -> dict:
        """Build a pyH2A input dict for blue hydrogen (SMR + CCS).

        Loads ``smr_blue_h2_params.md`` as a base template (merged with
        pyH2A defaults), then applies runtime overrides from the dataclass
        fields and the per-call price arguments.  This mirrors the pattern
        used by ``_build_green_dcf_input`` for the PV+E pathway.

        The markdown file defines the full plant structure — workflow,
        capital costs, CCS equipment, fixed O&M, and replacement schedule
        — while the code overrides spot prices, carbon prices, financial
        parameters, and any dataclass fields that differ from the file
        defaults.

        Parameters
        ----------
        ng_price_mwh : float
            Natural gas spot price in $/MWh.
        carbon_price : float
            Carbon price in $/tonne CO2.  Applied to uncaptured emissions.
        """
        # Load and cache the blue SMR base config (merged with pyH2A defaults).
        if not hasattr(self, '_blue_base_config'):
            self._blue_base_config = convert_input_to_dictionary(
                _BLUE_SMR_CONFIG
            )

        inp = copy.deepcopy(self._blue_base_config)

        # --- Workflow fix: remove Variable_Operating_Cost_Plugin that was
        # pulled in from pyH2A defaults; SMR_Natural_Gas_Plugin replaces it.
        inp['Workflow'].pop('Variable_Operating_Cost_Plugin', None)

        # --- Override financial parameters from dataclass ------------------
        fin = inp['Financial Input Values']
        fin['plant life']['Value']  = self.plant_life_years
        fin['irr']['Value']         = self.irr
        fin['equity']['Value']      = self.equity_fraction
        fin['inflation']['Value']   = self.inflation_rate
        fin['federal tax']['Value'] = self.federal_tax
        fin['state tax']['Value']   = self.state_tax

        # --- Override plant scale from dataclass ---------------------------
        tech = inp['Technical Operating Parameters and Specifications']
        tech['Plant Design Capacity (kg of H2/day)']['Value'] = self.smr_plant_capacity_kg_per_day
        tech['Operating Capacity Factor (%)']['Value']         = self.smr_capacity_factor

        # --- Override SMR + CCS parameters from dataclass ------------------
        smr = inp['Natural Gas SMR']
        smr['Spot Price ($/MWh)']['Value']           = ng_price_mwh
        smr['Carbon Price ($/tonne CO2)']['Value']   = carbon_price
        smr['Heat Rate (MWh/kg H2)']['Value']        = (
            self.smr_ng_consumption_mwh_per_kg * (1.0 + self.ccs_energy_penalty_frac)
        )
        smr['CO2 Intensity (kg CO2/kg H2)']['Value'] = self.smr_co2_intensity_kg_per_kg
        smr['CCS Capture Rate']['Value']             = self.ccs_capture_rate

        # --- Override capital costs from dataclass -------------------------
        inp['Direct Capital Cost - SMR']['SMR Plant Direct CAPEX ($)']['Value'] = (
            self.smr_direct_capex_usd + self.ccs_direct_capex_usd
        )
        inp['Indirect Capital Cost - SMR'][
            'SMR Indirect CAPEX (fraction of direct)'
        ]['Value'] = self.smr_indirect_capex_frac

        # --- Override CCS annual OPEX from dataclass -----------------------
        output_per_year = (
            self.smr_plant_capacity_kg_per_day * 365.0 * self.smr_capacity_factor
        )
        inp['Other Fixed Operating Cost - CCS']['CCS Annual OPEX ($)']['Value'] = (
            self.ccs_opex_per_kg * output_per_year
        )

        # --- Override fixed O&M from dataclass -----------------------------
        inp['Fixed Operating Costs']['staff']['Value']            = self.smr_staff
        inp['Fixed Operating Costs']['hourly labor cost']['Value'] = self.smr_hourly_labor_usd

        # --- Override land from dataclass ----------------------------------
        inp['Non-Depreciable Capital Costs']['Cost of land ($ per acre)']['Value'] = (
            self.smr_land_cost_per_acre
        )
        inp['Non-Depreciable Capital Costs']['Land required (acres)']['Value'] = (
            self.smr_land_acres
        )

        return inp

    def _build_green_dcf_input(self, power_price_mwh: float) -> dict:
        """Build a pyH2A input dict for PV + Electrolyzer green hydrogen.

        Uses the bundled ``210613_PV_E.md`` example as a base template,
        which chains ``Hourly_Irradiation_Plugin`` → ``Photovoltaic_Plugin``
        → ``Electrolyzer_Plugin`` for a solar-powered PEM system in
        Barstow, CA (TMY solar resource).

        The ``power_price_mwh`` input is mapped to PV panel CAPEX via an
        LCOE-equivalence scaling so that a higher electricity price implies
        proportionally more expensive photovoltaic capital::

            PV_CAPEX($/kW) = pv_reference_capex × (power_price / pv_reference_lcoe)

        The electrolyzer conversion efficiency is overridden from the
        dataclass field ``electrolyzer_consumption_mwh_per_kg``, and the
        shared financial parameters (IRR, equity, tax rates, inflation,
        plant life) are applied to the base case.

        Parameters
        ----------
        power_price_mwh : float
            Electricity spot price in $/MWh.  Mapped to PV CAPEX via
            LCOE equivalence.
        """
        # Load and cache the PV+E base config (merged with pyH2A defaults).
        if not hasattr(self, '_pv_e_base_config'):
            self._pv_e_base_config = convert_input_to_dictionary(
                _GREEN_SMR_CONFIG
            )

        inp = copy.deepcopy(self._pv_e_base_config)

        # --- Workflow fix: add Electrolyzer_Plugin between PV (pos 0) and
        # Production_Scaling_Plugin (pos 1).  The example omits it from its
        # Workflow table, but the Electrolyzer_Plugin is required to compute
        # Plant Design Capacity from the hourly PV power profile. ----------
        inp['Workflow']['Electrolyzer_Plugin'] = {
            'Type': 'plugin', 'Position': 0.5,
        }

        # --- Override financial parameters --------------------------------
        fin = inp['Financial Input Values']
        fin['plant life']['Value']  = self.plant_life_years
        fin['irr']['Value']         = self.irr
        fin['equity']['Value']      = self.equity_fraction
        fin['inflation']['Value']   = self.inflation_rate
        fin['federal tax']['Value'] = self.federal_tax
        fin['state tax']['Value']   = self.state_tax

        # --- Map power_price to PV CAPEX via LCOE equivalence -------------
        scaled_pv_capex = self.pv_reference_capex_per_kw * (
            power_price_mwh / self.pv_reference_lcoe_mwh
        )
        inp['Direct Capital Costs - PV']['PV CAPEX ($/kW)']['Value'] = scaled_pv_capex

        # --- Override electrolyzer conversion efficiency ------------------
        # Dataclass stores MWh/kg; pyH2A uses kg/kWh (inverse, ×1000).
        inp['Electrolyzer']['Conversion efficiency (kg H2/kWh)']['Value'] = (
            1.0 / (self.electrolyzer_consumption_mwh_per_kg * 1000.0)
        )

        return inp

    def _run_dcf(self, input_dict: dict) -> float:
        """Instantiate pyH2A Discounted_Cash_Flow and return LCOH in $/kg H2."""
        dcf = Discounted_Cash_Flow(
            input_dict,
            print_info=False,
            check_processing=False,
        )
        return float(dcf.h2_cost)

    # =================================================================== #
    #  Private scalar helpers (cached)                                    #
    # =================================================================== #

    def _grey_scalar(self, ng_price: float) -> float:
        key = (round(ng_price, 2), False, 0.0)
        if key not in self._smr_cache:
            self._smr_cache[key] = self._run_dcf(
                self._build_smr_dcf_input(ng_price, blue=False)
            )
        return self._smr_cache[key]

    def _blue_scalar(self, ng_price: float, carbon_price: float) -> float:
        key = (round(ng_price, 2), True, round(carbon_price, 1))
        if key not in self._smr_cache:
            self._smr_cache[key] = self._run_dcf(
                self._build_smr_dcf_input(ng_price, carbon_price, blue=True)
            )
        return self._smr_cache[key]

    def _green_scalar(self, power_price: float, carbon_price: float) -> float:
        # Carbon savings applied post-hoc to keep the DCF cache power-price-only.
        # Original formula: carbon_savings = carbon_price / 1000 ($/tCO2 → $/kgCO2,
        # implying a ~1 kg CO2 credit per kg H2 produced).
        power_key = round(power_price, 2)
        if power_key not in self._green_cache:
            self._green_cache[power_key] = self._run_dcf(
                self._build_green_dcf_input(power_price)
            )
        base_lcoh = self._green_cache[power_key]
        carbon_savings = float(carbon_price) / 1000.0
        return base_lcoh - carbon_savings

    # =================================================================== #
    #  LCOH computation methods (public, vectorised)                      #
    # =================================================================== #

    def compute_grey_lcoh(self, ng_price: np.ndarray | float) -> np.ndarray | float:
        """
        Grey hydrogen LCOH via unabated SMR, backed by pyH2A DCF.

        Args:
            ng_price: Natural gas price ($/MWh). Accepts a scalar float or a
                      numpy array of any shape — typically a (T, M) forward
                      curve or a (T,) spot vector.

        Returns:
            Grey H2 LCOH ($/kg), same type and shape as *ng_price*.
        """
        scalar_input = np.ndim(ng_price) == 0
        result = np.vectorize(self._grey_scalar, otypes=[float])(ng_price)
        return float(result) if scalar_input else result

    def compute_blue_lcoh(
        self,
        ng_price: np.ndarray | float,
        carbon_price: np.ndarray | float = 0.0,
    ) -> np.ndarray | float:
        """
        Blue hydrogen LCOH via SMR with CCS, backed by pyH2A DCF.

        The CCS energy penalty increases the effective gas consumption, and
        the CCS capital and OPEX are included as separate cost components.
        A carbon-savings term is embedded in the feedstock cost via the
        SMR_Natural_Gas_Plugin (carbon cost on uncaptured CO₂).

        Args:
            ng_price:     Natural gas price ($/MWh). Accepts a scalar float or
                          a numpy array of any shape.
            carbon_price: Carbon price ($/tCO2), scalar float or array
                          broadcastable to *ng_price*.  Defaults to 0.

        Returns:
            Blue H2 LCOH ($/kg), same type and shape as *ng_price*.
        """
        scalar_input = np.ndim(ng_price) == 0 and np.ndim(carbon_price) == 0
        result = np.vectorize(self._blue_scalar, otypes=[float])(
            ng_price, carbon_price
        )
        return float(result) if scalar_input else result

    def compute_green_lcoh(
        self,
        power_price: np.ndarray | float,
        carbon_price: np.ndarray | float = 0.0,
    ) -> np.ndarray | float:
        """
        Green hydrogen LCOH via PEM electrolysis, backed by pyH2A DCF.

        Args:
            power_price:  Electricity price ($/MWh). Accepts a scalar float or
                          a numpy array of any shape.
            carbon_price: Carbon price ($/tCO2), scalar float or array
                          broadcastable to *power_price*.  Defaults to 0.

        Returns:
            Green H2 LCOH ($/kg), same type and shape as *power_price*.
        """
        scalar_input = np.ndim(power_price) == 0 and np.ndim(carbon_price) == 0
        result = np.vectorize(self._green_scalar, otypes=[float])(
            power_price, carbon_price
        )
        return float(result) if scalar_input else result

    def compute_blue_green_h2_index(
        self,
        ng_price: np.ndarray,
        carbon_price: np.ndarray | float = 0.0,
        seasonality_alpha: float = 1.0,
        blue_contrib: float = BLUE_H2_CONTRIB,
        green_contrib: float = GREEN_H2_CONTRIB,
    ) -> np.ndarray:
        """
        Compute the Blue-Green H2 Index, defined as the average of blue and
        green LCOH converted to kEUR/tonne for benchmarking against energy curves:

            Blue-Green H2 Index = lcoh_per_kg_to_per_mwh(
                (LCOH_blue + LCOH_green) / 2
            )

        Args:
            ng_price: the complete forward curve of natural gas prices ($/MWh) as a numpy array of shape (T, M).
            carbon_price: the carbon price in $/tCO2, either as a scalar float or as a numpy array broadcastable to the shape of ng_price.
            seasonality_alpha: a scaling factor applied to the final index to reflect seasonal variations in hydrogen demand
                    or production costs. Defaults to 1.0 (no seasonality adjustment).
                    Must be within [0,1] to compress the existing seasonality given by the ng input curve.
        """
        blue  = self.compute_blue_lcoh(ng_price, carbon_price)
        green = self.compute_green_lcoh(PV_LCOE, carbon_price)

        # assume that 80% of index comes from blue and 20% comes from green
        blue_green_index = blue_contrib * blue + green_contrib * green

        # apply compression operation
        mean = np.mean(blue_green_index)
        compressed = mean + seasonality_alpha * (blue_green_index - mean)

        return compressed

    # =================================================================== #
    #  Unit conversion                                                    #
    # =================================================================== #

    @staticmethod
    def lcoh_per_kg_to_per_mwh(
        lcoh_per_kg: np.ndarray | float,
        heating_value_kwh_per_kg: float = 33.33,
    ) -> np.ndarray | float:
        """
        Convert hydrogen price from $/kg to $/MWh for direct benchmarking
        against natural gas and electricity forward curves.

        Formula::

            price_per_MWh = price_per_kg / (heating_value_kWh_per_kg / 1000)

        The default heating value is the Lower Heating Value (LHV) of hydrogen,
        33.33 kWh/kg (≈ 120 MJ/kg).  Energy markets conventionally quote on an
        LHV basis, matching the convention used for natural gas prices.  To
        benchmark on a Higher Heating Value (HHV) basis instead, pass
        ``heating_value_kwh_per_kg=39.41``.

        Key values:
            LHV (default) : 33.33 kWh/kg  (ISO 13443; widely used in EU gas markets)
            HHV           : 39.41 kWh/kg  (gross calorific value basis)

        References:
            IEA, "The Future of Hydrogen", 2019, Annex I.
            https://www.iea.org/reports/the-future-of-hydrogen

        Args:
            lcoh_per_kg:              H2 price or LCOH ($/kg). Accepts a scalar
                                      float or a numpy array of any shape.
            heating_value_kwh_per_kg: Energy content of hydrogen (kWh/kg).
                                      Defaults to LHV = 33.33 kWh/kg.

        Returns:
            H2 price ($/MWh), same type and shape as *lcoh_per_kg*.
        """
        return np.asarray(lcoh_per_kg) / (heating_value_kwh_per_kg / 1000.0)

    # =================================================================== #
    #  Convenience / introspection                                        #
    # =================================================================== #

    def summary(
        self,
        ng_price: float = 30.0,
        power_price: float = 50.0,
        carbon_price: float = 80.0,
    ) -> str:
        """
        Return a human-readable LCOH summary at given spot prices.

        Useful for quick sanity checks and thesis parameter tables.
        Note: first call at each price will run a full DCF; subsequent
        calls return cached results instantly.
        """
        grey  = self.compute_grey_lcoh(ng_price)
        blue  = self.compute_blue_lcoh(ng_price, carbon_price)
        green = self.compute_green_lcoh(power_price, carbon_price)

        lines = [
            "HydrogenLCOHCalculator — Point Estimates (pyH2A DCF)",
            f"  NG price     : {ng_price:.1f} $/MWh",
            f"  Power price  : {power_price:.1f} $/MWh",
            f"  Carbon price : {carbon_price:.1f} $/tCO2",
            "",
            f"  Grey  H2 LCOH : {grey:.2f} $/kg",
            f"  Blue  H2 LCOH : {blue:.2f} $/kg",
            f"  Green H2 LCOH : {green:.2f} $/kg",
            "",
            "Key plant constants (SMR):",
            f"  NG consumption  : {self.smr_ng_consumption_mwh_per_kg * 1000:.0f} kWh/kg H2",
            f"  CO2 intensity   : {self.smr_co2_intensity_kg_per_kg:.1f} kg CO2/kg H2",
            f"  Capacity        : {self.smr_plant_capacity_kg_per_day:,.0f} kg H2/day",
            f"  Direct CAPEX    : ${self.smr_direct_capex_usd:,.0f}",
            "",
            "Key plant constants (CCS):",
            f"  Capture rate    : {self.ccs_capture_rate * 100:.0f} %",
            f"  Energy penalty  : {self.ccs_energy_penalty_frac * 100:.0f} %",
            f"  Direct CAPEX    : ${self.ccs_direct_capex_usd:,.0f}",
            "",
            "Key plant constants (PEM):",
            f"  Electricity use : {self.electrolyzer_consumption_mwh_per_kg * 1000:.0f} kWh/kg H2",
            f"  Capacity        : {self.pem_plant_capacity_kg_per_day:,.0f} kg H2/day",
            f"  Direct CAPEX    : ${self.pem_direct_capex_usd:,.0f}",
        ]
        return "\n".join(lines)
