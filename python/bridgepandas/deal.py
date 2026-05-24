from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .hand import int_to_hand_str, hand_str_to_int, BridgeHandArray


@dataclass(frozen=True)
class Deal:
    """
    A single bridge deal: four hands stored as int64 bit fields.

    Immutable and hashable, so deals can be used as dict keys or in sets.
    Meant to be the unit of exchange between pandas (bulk storage) and
    bridge-logic code (bidding, double-dummy analysis, etc.).
    """

    west:  int
    north: int
    east:  int
    south: int

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_row(cls, row) -> Deal:
        """Create a Deal from a DataFrame row (e.g. ``df.iloc[i]``)."""
        return cls(
            west=int(row["west"]),
            north=int(row["north"]),
            east=int(row["east"]),
            south=int(row["south"]),
        )

    @classmethod
    def from_strings(cls, west: str, north: str, east: str, south: str) -> Deal:
        """Create a Deal from four S/H/D/C hand strings."""
        return cls(
            west=hand_str_to_int(west),
            north=hand_str_to_int(north),
            east=hand_str_to_int(east),
            south=hand_str_to_int(south),
        )

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, int]:
        return {"west": self.west, "north": self.north,
                "east": self.east, "south": self.south}

    @staticmethod
    def to_dataframe(deals: Iterable[Deal]) -> pd.DataFrame:
        """Convert an iterable of Deals to a DataFrame with BridgeHandArray columns."""
        cols: dict[str, list[int]] = {"west": [], "north": [], "east": [], "south": []}
        for deal in deals:
            cols["west"].append(deal.west)
            cols["north"].append(deal.north)
            cols["east"].append(deal.east)
            cols["south"].append(deal.south)
        return pd.DataFrame({
            name: BridgeHandArray(np.array(vals, dtype=np.int64))
            for name, vals in cols.items()
        })

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        w = int_to_hand_str(self.west)
        n = int_to_hand_str(self.north)
        e = int_to_hand_str(self.east)
        s = int_to_hand_str(self.south)
        width = max(len(w), len(s))
        pad = " " * (width + 3)
        return (
            f"{pad}N  {n}\n"
            f"W  {w:{width}}   E  {e}\n"
            f"{pad}S  {s}"
        )
