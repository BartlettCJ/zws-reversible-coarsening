"""ZWSLab.

A research package for exact, local, reversible, signed-sum-preserving
cellular automata and adversarial finite-window analysis.
"""

from .alphabet import Alphabet
from .rules import ReversibleRule

__all__ = ["Alphabet", "ReversibleRule"]
__version__ = "0.1.4"
