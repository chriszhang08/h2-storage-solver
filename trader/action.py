from dataclasses import dataclass

from constants import STORAGE_CAPACITY, WITHDRAWAL_COST_PER_UNIT, INJECTION_COST_PER_UNIT, BOIL_OFF


@dataclass
class InterpretedAction:
    # -- Input parameters (required) --
    discrete_action: int  # 0 = withdraw, 1 = do nothing, 2 = inject

    # the real h2 spot differs from the market h2 spot because of per unit storage constraints
    real_h2_spot: float = 0.0

    # constraints
    h2_inventory: float = 0.0
    max_withdraw_rate: float = 0.0
    max_inject_rate: float = 0.0

    # -- Computed quantities --
    h2_withdraw_units: float = 0.0
    h2_withdraw_dollars: float = 0.0
    h2_inject_units: float = 0.0
    h2_inject_dollars: float = 0.0

    boil_off_units: float = 0.0

    def __post_init__(self):
        # compute boil off units
        self.boil_off_units = self.h2_inventory * BOIL_OFF

        if self.discrete_action == 0:
            # include the WITHDRAWAL_COST_PER_UNIT in the spot price
            self.real_h2_spot -= WITHDRAWAL_COST_PER_UNIT

            # Withdraw at max rate, capped by current inventory
            desired_withdrawal = self.max_withdraw_rate
            self.h2_withdraw_units = min(desired_withdrawal, self.h2_inventory)
            self.h2_withdraw_dollars = self.h2_withdraw_units * self.real_h2_spot

        elif self.discrete_action == 2:
            # include the INJECTION_COST_PER_UNIT in the spot price
            self.real_h2_spot += INJECTION_COST_PER_UNIT

            # Inject at max rate, capped by remaining storage capacity
            desired_injection = self.max_inject_rate
            remaining_capacity = STORAGE_CAPACITY - self.h2_inventory
            self.h2_inject_units = min(desired_injection, remaining_capacity)
            self.h2_inject_dollars = self.h2_inject_units * self.real_h2_spot

        # action == 1: do nothing — all quantities remain 0.0

    @classmethod
    def interpret(
        cls,
        action: int,
        curr_h2_spot: float,
        h2_inventory: float,
        max_withdraw_rate: float,
        max_inject_rate: float,
    ) -> "InterpretedAction":
        """
        Interpret a discrete action into constrained injection/withdrawal decisions.

        :param action: (int): 0 = withdraw at max rate, 1 = do nothing, 2 = inject at max rate.
        :param curr_h2_spot: (float): Current spot price of H2 in $/kg.
        :param h2_inventory: (float): Current H2 inventory in kg.
        :param max_withdraw_rate: (float): Maximum units that can be withdrawn in a single step.
        :param max_inject_rate: (float): Maximum units that can be injected in a single step.

        :return: (InterpretedAction): The human-readable action object with computed injection/withdrawal units and dollars.
        """
        return cls(
            discrete_action=int(action),
            real_h2_spot=curr_h2_spot,
            h2_inventory=h2_inventory,
            max_withdraw_rate=max_withdraw_rate,
            max_inject_rate=max_inject_rate,
        )

    def __repr__(self) -> str:
        action_names = {0: "withdraw", 1: "nothing", 2: "inject"}
        return (
            f"InterpretedAction(action={action_names.get(self.discrete_action, '?')}, "
            f"h2_withdraw_units={self.h2_withdraw_units:.2f}, h2_inject_units={self.h2_inject_units:.2f})"
        )

    def to_dict(self) -> dict:
        return {
            "discrete_action": self.discrete_action,
            "real_h2_spot": self.real_h2_spot,
            "h2_inventory": self.h2_inventory,
            "max_withdraw_rate": self.max_withdraw_rate,
            "max_inject_rate": self.max_inject_rate,
            "h2_withdraw_units": self.h2_withdraw_units,
            "h2_withdraw_dollars": self.h2_withdraw_dollars,
            "h2_inject_units": self.h2_inject_units,
            "h2_inject_dollars": self.h2_inject_dollars,
            "boil_off_units": self.boil_off_units,
        }
