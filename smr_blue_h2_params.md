# Workflow

Name | Type | Position
--- | --- | ---
SMR_Natural_Gas_Plugin | plugin | 10

# Financial Input Values

Name | Full Name | Value
--- | --- | ---
ref year | Reference year | 2024
startup year | Assumed start-up year | 2024
basis year | Basis year | 2024
current year capital costs | Current year for capital costs | 2024
startup time | Start-up Time (years) | 1
plant life | Plant life (years) | 20
equity | % Equity Financing | 40%
irr | After-tax Real IRR (%) | 8.0%
inflation | Inflation rate (%) | 1.9%
state tax | State Taxes (%) | 6.0%
federal tax | Federal Taxes (%) | 21.0%

# Construction

Name | Full Name | Value
--- | --- | ---
capital perc 1st | % of Capital Spent in 1st Year of Construction | 100%

# Technical Operating Parameters and Specifications

Name | Value
--- | ---
Plant Design Capacity (kg of H2/day) | 50,000
Operating Capacity Factor (%) | 90%

# Natural Gas SMR

Name | Value
--- | ---
Spot Price ($/MWh) | 0
Heat Rate (MWh/kg H2) | 0.0528
CO2 Intensity (kg CO2/kg H2) | 9.0
CCS Capture Rate | 90%
Carbon Price ($/tonne CO2) | 0

# Direct Capital Cost - SMR

Name | Value
--- | ---
SMR Plant Direct CAPEX ($) | 280,000,000

# Indirect Capital Cost - SMR

Name | Value | Path
--- | --- | ---
SMR Indirect CAPEX (fraction of direct) | 33% | Direct Capital Cost - SMR > SMR Plant Direct CAPEX ($) > Value

# Non-Depreciable Capital Costs

Name | Value
--- | ---
Cost of land ($ per acre) | 10,000
Land required (acres) | 5

# Fixed Operating Costs

Name | Full Name | Value
--- | --- | ---
staff | Operating staff headcount | 15
hourly labor cost | Burdened labor cost ($/hr) | 50.0

# Other Fixed Operating Cost - CCS

Name | Value
--- | ---
CCS Annual OPEX ($) | 7,719,750

# Planned Replacement

Name | Cost ($) | Frequency (years)
--- | --- | ---
