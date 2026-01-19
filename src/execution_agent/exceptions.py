# execution_agent/exceptions.py
from __future__ import annotations


class GoalsAccomplished(Exception):
    """Raised to stop the run loop cleanly when goals_accomplished is called."""
    pass


class FormatError(Exception):
    """Raised when the model output is not valid / not parseable per our contract."""
    pass


class BudgetExhausted(Exception):
    """Raised when agent exhausts step budget without accomplishing goals."""
    pass
