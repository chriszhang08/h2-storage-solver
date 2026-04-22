NUM_M = 36  # spot, M1, M2, ... M36
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_MONTH = TRADING_DAYS_PER_YEAR / 12
KNOT_VECTOR = [1, 6, 12, 24]
Z_SCORE_CLIP = 8

STORAGE_CAPACITY: float = 1000.0  # Max H2 inventory capacity in tonnes
EPS: float = 1e-8  # Small constant to prevent division by zero in reward calculations

H2_STORAGE_CAPACITY: float = (
    1000.0  # TODO: Max H2 storage capacity (kg) — set from domain literature
)

CH4_INJCT_WTHDRW_COST: float = (
    0.01  # Cost per kg of CH4 injected into storage (EUR/MWh) — set from domain literature
)
H2_INJCT_WTHDRW_COST: float = (
    0.02  # Cost per kg of H2 injected into storage (EUR/MWh) — set from domain literature
)

LOHC_MAX_INJECTION_RATE: float = (
    40.0  # Max injection rate for LOHC-based storage (tonnes per step) — set from domain literature
)
LOHC_MAX_WITHDRAW_RATE: float = (
    40.0  # Max withdrawal rate for LOHC-based storage (tonnes per step) — set from domain literature
)

SMR_EFFICIENCY: float = (
    0.7
)

CCUS_COST_PER_TON: float = (
    80.0  # Carbon capture cost per ton of CO2 (EUR) — set from domain literature
)

H2_SPECIFIC_HEAT: float = (
    33.33  # Specific heat of H2 in kWh/kg (LHV basis) — set from domain literature
)
PV_LCOE: float = (
    30.0  # Levelized cost of electricity from PV in EUR/MWh — set from domain literature
)

# the fraction contribution of green hydrogen to the h2 price index
GREEN_H2_CONTRIB = 0.2

# the fractional contribution of blue hydrogen to the h2 price index
BLUE_H2_CONTRIB = 0.8
