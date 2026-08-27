"""Synthetic query generation strategies, keyed by the paper each reimplements."""

from __future__ import annotations

from generation.dragon import DragonGenerator
from generation.duqgen import DuqgenGenerator
from generation.inpars import InParsGenerator
from generation.naive import NaiveGenerator
from generation.promptagator import PromptagatorGenerator
from generation.udapdr import UdapdrGenerator

STRATEGIES: dict[str, type] = {
    "naive": NaiveGenerator,
    "inpars": InParsGenerator,
    "promptagator": PromptagatorGenerator,
    "duqgen": DuqgenGenerator,
    "udapdr": UdapdrGenerator,
    "dragon": DragonGenerator,
}

__all__ = [
    "STRATEGIES",
    "NaiveGenerator",
    "InParsGenerator",
    "PromptagatorGenerator",
    "DuqgenGenerator",
    "UdapdrGenerator",
    "DragonGenerator",
]
