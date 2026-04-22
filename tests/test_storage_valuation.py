"""
Tests for storage_valuation() in trader/rewards/reward_calc_utils.py.

Storage valuation solves an LP over an interpolated forward curve
to find the optimal inject/withdraw schedule that maximises risk-neutral NPV,
subject to:
  - inventory floor  >= 0
  - inventory cap    <= STORAGE_CAPACITY
  - terminal fill    == end_inventory  (default: 0, i.e. empty at horizon)

NOTE on feasibility
-------------------
STORAGE_CAPACITY = 1000 units.  The default max_inject_rate / max_withdraw_rate
is 1.0 unit/month.  The LP decision variable per granularity step is bounded by
max_rate * (TRADING_DAYS_PER_MONTH / granularity_of_action_per_month).
With default granularity=4, that is max_rate * 5.25 units/step.
A 12-month horizon has 12 * 4 = 48 steps, so at rate=1.0 the maximum cumulative
injection is 48 * 5.25 = 252 << 1000, making most tests infeasible.

We therefore use HIGH_RATE (= STORAGE_CAPACITY / 50 = 20) as the default rate for
tests that are not specifically testing rate-limit behaviour.  At HIGH_RATE=20:
  per-step bound = 20 * 5.25 = 105 units/step
  48-step max   = 48 * 105   = 5040 >> 1000  (always feasible)

Rate-bound assertions must compare against the *per-step* bound, not the raw rate:
  per_step_bound = max_rate * (TRADING_DAYS_PER_MONTH / DEFAULT_GRANULARITY)
"""

import numpy as np
import pytest

from constants import STORAGE_CAPACITY, TRADING_DAYS_PER_MONTH
from trader.rewards.reward_calc_utils import storage_valuation
from analysis.visualizers.reward_visualizer import StorageValuationVisualizer


# ---------------------------------------------------------------------------
# Module-level parameters
# ---------------------------------------------------------------------------

DEFAULT_GRANULARITY: int = 4  # granularity_of_action_per_month default

# A high rate that makes every test scenario feasible from inventory=0
HIGH_RATE: float = (
    STORAGE_CAPACITY / 20
)  # 20 units/month => 105 units/step at granularity=4


# Per-step bound corresponding to a given monthly rate (at default granularity)
def per_step_bound(rate: float, granularity: int = DEFAULT_GRANULARITY) -> float:
    return rate * (TRADING_DAYS_PER_MONTH / granularity)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def flat_curve(price: float = 10.0, n_months: int = 12) -> np.ndarray:
    return np.full(n_months, price)


def ascending_curve(
    start: float = 5.0, end: float = 20.0, n_months: int = 12
) -> np.ndarray:
    return np.linspace(start, end, n_months)


def descending_curve(
    start: float = 20.0, end: float = 5.0, n_months: int = 12
) -> np.ndarray:
    return np.linspace(start, end, n_months)


def call(curve, inventory=0.0, **kwargs):
    """Convenience wrapper that defaults to HIGH_RATE for inject/withdraw."""
    kwargs.setdefault("max_inject_rate", HIGH_RATE)
    kwargs.setdefault("max_withdraw_rate", HIGH_RATE)
    return storage_valuation(curve, inventory, **kwargs)


# ---------------------------------------------------------------------------
# Return-structure tests
# ---------------------------------------------------------------------------


class TestReturnStructure:
    """The function must return a dict with all documented keys."""

    REQUIRED_KEYS = {
        "npv",
        "action_schedule",
        "inject_schedule",
        "withdraw_schedule",
        "inventory_path",
        "revenue_schedule",
        "purchase_cost_schedule",
        "net_cashflow_schedule",
        "discounted_cashflows",
        "discount_factors",
    }

    def test_all_keys_present(self):
        result = call(flat_curve(), inventory=0.0)
        assert self.REQUIRED_KEYS == set(result.keys())

    def test_npv_is_scalar_float(self):
        result = call(flat_curve(), inventory=0.0)
        assert isinstance(result["npv"], float)

    def test_array_lengths_match(self):
        n_months = 6
        result = call(flat_curve(n_months=n_months), inventory=0.0)
        T = len(result["action_schedule"])
        for key in [
            "inject_schedule",
            "withdraw_schedule",
            "inventory_path",
            "revenue_schedule",
            "purchase_cost_schedule",
            "net_cashflow_schedule",
            "discounted_cashflows",
            "discount_factors",
        ]:
            assert len(result[key]) == T, f"{key} has wrong length"


# ---------------------------------------------------------------------------
# Physical constraint tests
# ---------------------------------------------------------------------------


class TestPhysicalConstraints:
    """Inventory must stay within [0, STORAGE_CAPACITY] at every step."""

    def test_inventory_never_negative(self):
        result = call(ascending_curve(), inventory=0.0)
        assert np.all(result["inventory_path"] >= -1e-6), "Inventory went below zero"

    def test_inventory_never_exceeds_capacity(self):
        result = call(ascending_curve(), inventory=0.0)
        assert np.all(
            result["inventory_path"] <= STORAGE_CAPACITY + 1e-6
        ), "Inventory exceeded storage capacity"

    def test_terminal_inventory_equals_end_inventory(self):
        """LP terminal constraint: inventory must equal end_inventory at horizon."""
        end_inv = STORAGE_CAPACITY
        result = call(flat_curve(), inventory=0.0, end_inventory=end_inv)
        assert (
            abs(result["inventory_path"][-1] - end_inv) < 1e-4
        ), f"Terminal inventory did not reach end_inventory={end_inv}"

    def test_terminal_inventory_default_is_zero(self):
        """Default end_inventory=0: tank must be empty at horizon."""
        result = call(flat_curve(), inventory=STORAGE_CAPACITY)
        assert (
            abs(result["inventory_path"][-1]) < 1e-4
        ), "Default terminal inventory should be 0"

    def test_terminal_inventory_with_partial_start(self):
        """Custom end_inventory is respected regardless of starting level."""
        initial = STORAGE_CAPACITY / 2
        end_inv = STORAGE_CAPACITY
        result = call(flat_curve(), inventory=initial, end_inventory=end_inv)
        assert abs(result["inventory_path"][-1] - end_inv) < 1e-4

    def test_inject_withdraw_are_nonnegative(self):
        result = call(ascending_curve(), inventory=0.0)
        assert np.all(result["inject_schedule"] >= -1e-9)
        assert np.all(result["withdraw_schedule"] >= -1e-9)

    def test_inject_withdraw_respect_max_rates(self):
        """Per-step volumes must not exceed the scaled per-step bound."""
        max_rate = HIGH_RATE / 2  # still feasible; just half the capacity per step
        step_bound = per_step_bound(max_rate)
        result = storage_valuation(
            flat_curve(),
            inventory=STORAGE_CAPACITY,
            end_inventory=0.0,
            max_inject_rate=max_rate,
            max_withdraw_rate=max_rate,
        )
        assert np.all(
            result["inject_schedule"] <= step_bound + 1e-6
        ), f"inject_schedule exceeded per-step bound {step_bound}"
        assert np.all(
            result["withdraw_schedule"] <= step_bound + 1e-6
        ), f"withdraw_schedule exceeded per-step bound {step_bound}"

    def test_action_equals_inject_minus_withdraw(self):
        """action[t] == inject[t] - withdraw[t] by construction."""
        result = call(ascending_curve(), inventory=0.0)
        reconstructed = result["inject_schedule"] - result["withdraw_schedule"]
        np.testing.assert_allclose(
            result["action_schedule"],
            reconstructed,
            atol=1e-6,
            err_msg="action != inject - withdraw",
        )


# ---------------------------------------------------------------------------
# Economic property tests
# ---------------------------------------------------------------------------


class TestEconomicProperties:
    """Verify qualitative economic behaviour of the LP solution."""

    def test_npv_scales_linearly_with_price(self):
        """NPV is linear in prices (LP objective is linear), so doubling all
        prices must double the NPV."""
        base = ascending_curve(start=5.0, end=10.0)
        r1 = call(base, inventory=0.0)
        r2 = call(base * 2, inventory=0.0)
        np.testing.assert_allclose(
            r2["npv"],
            2 * r1["npv"],
            rtol=1e-2,
            err_msg="NPV should scale linearly with price",
        )

    def test_ascending_curve_only_injects_early(self):
        """On a strictly rising curve, rational strategy = buy early, sell late.
        Net actions should be positive (inject) in the first half and negative
        (withdraw) later — injections must dominate the first half."""
        result = call(ascending_curve(start=1.0, end=20.0), inventory=0.0)
        T = len(result["action_schedule"])
        first_half_net = result["action_schedule"][: T // 2].sum()
        second_half_net = result["action_schedule"][T // 2 :].sum()
        assert (
            first_half_net > second_half_net
        ), "Expected net injection in first half and net withdrawal in second half"

    def test_zero_discount_rate_flat_price_npv_is_zero(self):
        """With zero discounting, a flat price, and a round-trip (start == end),
        there is no time-value arbitrage: NPV == 0."""
        result = storage_valuation(
            flat_curve(price=10.0),
            inventory=STORAGE_CAPACITY,
            end_inventory=STORAGE_CAPACITY,  # round trip: no net obligation
            discount_rate=0.0,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
        )
        assert (
            abs(result["npv"]) < 1e-4
        ), f"Zero-discount flat-curve round-trip NPV should be 0, got {result['npv']:.6f}"

    def test_higher_initial_inventory_does_not_reduce_npv(self):
        """More inventory at the start gives at least as much optionality."""
        res_empty = call(ascending_curve(), inventory=0.0)
        res_half = call(ascending_curve(), inventory=STORAGE_CAPACITY / 2)
        assert (
            res_half["npv"] >= res_empty["npv"] - 1e-4
        ), "Higher starting inventory should not reduce NPV"

    def test_revenue_and_cost_are_nonneg(self):
        result = call(ascending_curve(), inventory=0.0)
        assert np.all(result["revenue_schedule"] >= -1e-9)
        assert np.all(result["purchase_cost_schedule"] >= -1e-9)

    def test_net_cashflow_equals_revenue_minus_cost(self):
        result = call(flat_curve(), inventory=0.0)
        expected = result["revenue_schedule"] - result["purchase_cost_schedule"]
        np.testing.assert_allclose(result["net_cashflow_schedule"], expected, atol=1e-9)

    def test_discounted_cashflows_sum_equals_npv(self):
        result = call(flat_curve(), inventory=0.0)
        np.testing.assert_allclose(
            result["discounted_cashflows"].sum(), result["npv"], atol=1e-6
        )

    def test_discount_factors_are_decreasing(self):
        """df[t] = 1/(1+r/12)^(t+1) is strictly decreasing."""
        result = call(flat_curve(), inventory=0.0)
        df = result["discount_factors"]
        assert np.all(np.diff(df) < 0), "Discount factors should be strictly decreasing"

    def test_discount_factors_are_positive_and_leq_one(self):
        result = call(flat_curve(), inventory=0.0)
        df = result["discount_factors"]
        assert np.all(df > 0) and np.all(df <= 1.0)


# ---------------------------------------------------------------------------
# Edge-case / robustness tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Corner cases and parameter boundary conditions."""

    def test_already_full_inventory_round_trip(self):
        """Starting fully stocked with end_inventory=STORAGE_CAPACITY: terminal
        constraint is met by doing nothing (or arbitrary round trips)."""
        result = storage_valuation(
            flat_curve(),
            inventory=STORAGE_CAPACITY,
            end_inventory=STORAGE_CAPACITY,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
        )
        assert isinstance(result["npv"], float)
        assert abs(result["inventory_path"][-1] - STORAGE_CAPACITY) < 1e-4

    def test_single_month_curve(self):
        """A one-month forward curve should still produce a feasible result."""
        result = storage_valuation(
            np.array([10.0]),
            inventory=0.0,
            end_inventory=0.0,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
        )
        assert isinstance(result["npv"], float)

    def test_zero_discount_rate(self):
        result = call(ascending_curve(), inventory=0.0, discount_rate=0.0)
        assert isinstance(result["npv"], float)
        # With rate=0, df[t] = 1/(1+0)^(t+1) = 1 for all t
        np.testing.assert_allclose(result["discount_factors"], 1.0, atol=1e-9)

    def test_high_discount_rate(self):
        result = call(ascending_curve(), inventory=0.0, discount_rate=0.50)
        assert isinstance(result["npv"], float)

    def test_tight_rates_feasible_over_long_horizon(self):
        """Low per-month rate still feasible when the horizon is long enough.
        With granularity=4 and rate=1.0:
          per-step bound = 1.0 * (21/4) = 5.25 units
          36 months * 4 steps = 144 steps; max inject = 144 * 5.25 = 756 < 1000.
        So we use n_months=48 for guaranteed feasibility (48*4*5.25=1008 > 1000)."""
        result = storage_valuation(
            flat_curve(n_months=48),
            inventory=0.0,
            end_inventory=0.0,
            max_inject_rate=1.0,
            max_withdraw_rate=1.0,
        )
        assert isinstance(result["npv"], float)
        step_bound = per_step_bound(1.0)
        assert np.all(result["inject_schedule"] <= step_bound + 1e-6)
        assert np.all(result["withdraw_schedule"] <= step_bound + 1e-6)

    def test_reproducibility(self):
        """Calling with the same inputs twice should return the identical NPV."""
        curve = ascending_curve()
        r1 = call(curve, inventory=0.0)
        r2 = call(curve, inventory=0.0)
        assert abs(r1["npv"] - r2["npv"]) < 1e-9

    def test_inventory_path_consistency(self):
        """inventory_path[t] == initial_inventory + cumsum(action)[t]."""
        initial = 100.0
        result = call(ascending_curve(), inventory=initial)
        expected_path = initial + np.cumsum(result["action_schedule"])
        np.testing.assert_allclose(result["inventory_path"], expected_path, atol=1e-6)

    def test_infeasible_raises_runtime_error(self):
        """An LP that cannot satisfy the terminal fill constraint must raise."""
        # end_inventory=STORAGE_CAPACITY but rate is far too low to fill in 1 month
        with pytest.raises(
            RuntimeError, match="LP solver failed: Infeasible"
        ):
            storage_valuation(
                flat_curve(n_months=1),
                inventory=0.0,
                end_inventory=STORAGE_CAPACITY * 2,
                max_inject_rate=0.1,
                max_withdraw_rate=0.1,
            )


# ---------------------------------------------------------------------------
# injection_allowed parameter tests
# ---------------------------------------------------------------------------


class TestInjectionAllowed:
    """Tests for the injection_allowed=False mode (withdraw-only LP)."""

    def test_no_injections_when_disabled(self):
        """inject_schedule must be all zeros when injection_allowed=False."""
        result = storage_valuation(
            descending_curve(),
            inventory=STORAGE_CAPACITY,
            end_inventory=0.0,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
            injection_allowed=False,
        )
        np.testing.assert_allclose(
            result["inject_schedule"],
            0.0,
            atol=1e-6,
            err_msg="inject_schedule should be zero when injection_allowed=False",
        )

    def test_actions_nonpositive_when_disabled(self):
        """All action values must be <= 0 (withdraw-only) when injection_allowed=False."""
        result = storage_valuation(
            descending_curve(),
            inventory=STORAGE_CAPACITY,
            end_inventory=0.0,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
            injection_allowed=False,
        )
        assert np.all(
            result["action_schedule"] <= 1e-6
        ), "action_schedule should have no positive values when injection_allowed=False"

    def test_feasible_when_inventory_sufficient(self):
        """injection_allowed=False is feasible when inventory >= end_inventory."""
        result = storage_valuation(
            ascending_curve(),
            inventory=STORAGE_CAPACITY,
            end_inventory=0.0,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
            injection_allowed=False,
        )
        assert isinstance(result["npv"], float)
        assert abs(result["inventory_path"][-1]) < 1e-4

    def test_infeasible_when_must_inject(self):
        """injection_allowed=False must raise when end_inventory > inventory
        (i.e. filling is required but injections are forbidden)."""
        with pytest.raises(
            RuntimeError, match="LP solver failed: Infeasible"
        ):
            storage_valuation(
                ascending_curve(),
                inventory=0.0,
                end_inventory=STORAGE_CAPACITY,  # requires injection
                max_inject_rate=HIGH_RATE,
                max_withdraw_rate=HIGH_RATE,
                injection_allowed=False,
            )

    def test_npv_with_injection_disabled_leq_enabled(self):
        """Restricting injections can only reduce (or equal) the unconstrained NPV."""
        curve = ascending_curve(start=1.0, end=20.0)
        res_full = storage_valuation(
            curve,
            inventory=STORAGE_CAPACITY,
            end_inventory=0.0,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
            injection_allowed=True,
        )
        res_withdraw_only = storage_valuation(
            curve,
            inventory=STORAGE_CAPACITY,
            end_inventory=0.0,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
            injection_allowed=False,
        )
        assert (
            res_withdraw_only["npv"] <= res_full["npv"] + 1e-6
        ), "Disabling injections should not increase NPV"

    def test_physical_constraints_still_hold_when_disabled(self):
        """Inventory bounds must be respected even in withdraw-only mode."""
        result = storage_valuation(
            descending_curve(),
            inventory=STORAGE_CAPACITY / 2,
            end_inventory=0.0,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
            injection_allowed=False,
        )
        assert np.all(
            result["inventory_path"] >= -1e-6
        ), "Inventory went below zero in withdraw-only mode"
        assert np.all(
            result["inventory_path"] <= STORAGE_CAPACITY + 1e-6
        ), "Inventory exceeded capacity in withdraw-only mode"

    def test_inject_withdraw_identity_holds_when_disabled(self):
        """action == inject - withdraw still holds when injection_allowed=False."""
        result = storage_valuation(
            descending_curve(),
            inventory=STORAGE_CAPACITY,
            end_inventory=0.0,
            max_inject_rate=HIGH_RATE,
            max_withdraw_rate=HIGH_RATE,
            injection_allowed=False,
        )
        reconstructed = result["inject_schedule"] - result["withdraw_schedule"]
        np.testing.assert_allclose(
            result["action_schedule"],
            reconstructed,
            atol=1e-6,
            err_msg="action != inject - withdraw in withdraw-only mode",
        )

    def test_enabled_by_default(self):
        """injection_allowed defaults to True; LP can freely inject."""
        result_default = call(ascending_curve(), inventory=0.0)
        result_explicit = call(ascending_curve(), inventory=0.0, injection_allowed=True)
        np.testing.assert_allclose(
            result_default["npv"],
            result_explicit["npv"],
            atol=1e-9,
            err_msg="Default injection_allowed=True should match explicit True",
        )


class TestSpecificCases:
    def test_contango_curve_npv_valuation(self):
        """On a contango curve, the LP should find a positive NPV by buying low and selling high."""
        fwd_curve = [
            95.41273445966885,
            102.58760076534408,
            108.52462162075243,
            115.90524555606154,
            130.4368628552613,
            134.8730429650437,
            135.3622396047128,
            146.44362726806668,
            142.03392493534724,
            120.84855516631266,
            115.80923417087838,
            119.45467782448074,
            116.95316230637867,
            118.0872572640173,
            119.83227326031977,
            124.3505549969371,
            136.83038905770698,
            139.5062411425221,
            138.10760669424624,
            147.1552846306906,
            140.6477639434074,
            117.38223694035148,
            110.3621898033711,
            112.20810849372327,
            108.15641071708069,
            108.00329442888453,
            108.7240702387226,
            112.48108278678583,
            124.46261872381614,
            126.90314376996349,
            125.53215324090225,
            134.8704461564846,
            128.91651157979578,
            106.46754162720941,
            100.527022512729,
            103.71544034365479,
        ]

        # convert to np array and call valuation
        fwd_curve = np.array(fwd_curve)
        result = call(fwd_curve, inventory=50.0, end_inventory=0.0)
        result_2 = call(fwd_curve, inventory=0.0, end_inventory=0.0)

        # Visualise the LP output
        viz = StorageValuationVisualizer(
            result,
            fwd_curve=fwd_curve,
            title_suffix="(contango test curve, 36 months)",
        )
        viz.generate_diagnostic_suite(
            output_dir="analysis/lp_debug/",
            show=True,
            filename="contango_lp_diagnostic.png",
        )

        assert result["npv"] > 0.0, "Expected positive NPV on a contango curve"
