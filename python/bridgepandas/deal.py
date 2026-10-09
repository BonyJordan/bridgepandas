from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .hand import Hand, int_to_hand_str, hand_str_to_int, BridgeHandArray, RANKS, _RANK_INDEX, _SUIT_OFFSET

# suits in bit-position order: bits 0-12 = C, 13-25 = D, 26-38 = H, 39-51 = S
_SUITS_BY_OFFSET = "CDHS"


class Deal:
    """
    A single bridge deal: four hands stored as Hand (int subclass) fields.

    Immutable and hashable, so deals can be used as dict keys or in sets.
    Each hand is accessible as an attribute, a full lowercase name, a
    single-letter direction, or a Direction object: ``deal.west``,
    ``deal["west"]``, ``deal["W"]``, or ``deal[direction]``.
    Printing a hand shows the S/H/D/C hand string.
    """

    __slots__ = ("west", "north", "east", "south")

    def __init__(self, west, north=None, east=None, south=None):
        """
        Two calling forms:

        ``Deal(west, north, east, south)`` — four hands as Hand, int, or hand string.

        ``Deal(row)`` — a single DataFrame row (from ``df.itertuples()`` or
        ``df.iloc[i]``); the row must have ``west``, ``north``, ``east``, ``south``
        attributes or keys.  Prefer :meth:`from_row` for clarity.
        """
        if north is None:
            row = west
            try:
                west, north, east, south = row.west, row.north, row.east, row.south
            except AttributeError:
                west, north, east, south = row["west"], row["north"], row["east"], row["south"]
        object.__setattr__(self, "west",  Hand(west))
        object.__setattr__(self, "north", Hand(north))
        object.__setattr__(self, "east",  Hand(east))
        object.__setattr__(self, "south", Hand(south))

    _KEY_MAP = {"W": "west", "N": "north", "E": "east", "S": "south",
                "west": "west", "north": "north", "east": "east", "south": "south"}

    def __getitem__(self, key):
        from .direction import Direction
        if isinstance(key, Direction):
            key = str(key)
        attr = Deal._KEY_MAP.get(key)
        if attr is None:
            raise KeyError(f"Invalid direction {key!r}; use 'W', 'N', 'E', or 'S'")
        return getattr(self, attr)

    def __setattr__(self, name, value):
        raise AttributeError("Deal is immutable")

    def __eq__(self, other):
        if not isinstance(other, Deal):
            return NotImplemented
        return (self.west == other.west and self.north == other.north
                and self.east == other.east and self.south == other.south)

    def __hash__(self):
        return hash((int(self.west), int(self.north), int(self.east), int(self.south)))

    def __repr__(self) -> str:
        return (f"Deal(west={str(self.west)!r}, north={str(self.north)!r}, "
                f"east={str(self.east)!r}, south={str(self.south)!r})")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_row(cls, row) -> Deal:
        """Create a Deal from a DataFrame row — supports iloc[] rows and itertuples() rows."""
        return cls(row)

    @classmethod
    def from_strings(cls, west: str, north: str, east: str, south: str) -> Deal:
        """Create a Deal from four S/H/D/C hand strings."""
        return cls(west, north, east, south)

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Hand]:
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
        w = str(self.west)
        n = str(self.north)
        e = str(self.east)
        s = str(self.south)
        width = max(len(w), len(s))
        pad = " " * (width + 3)
        return (
            f"{pad}N  {n}\n"
            f"W  {w:{width}}   E  {e}\n"
            f"{pad}S  {s}"
        )


# ---------------------------------------------------------------------------
# random_deals
# ---------------------------------------------------------------------------

def _parse_partial_hand(s: str) -> int:
    """Parse a partial hand string (S/H/D/C, fewer than 13 cards ok) to int64 bitmask."""
    parts = s.split("/")
    if len(parts) != 4:
        raise ValueError(f"Expected 4 suit groups separated by '/': {s!r}")
    result = 0
    for suit, ranks in zip("SHDC", parts):
        offset = _SUIT_OFFSET[suit]
        if ranks == "-":
            continue
        for ch in ranks:
            if ch not in _RANK_INDEX:
                raise ValueError(f"Unknown rank {ch!r} in {s!r}")
            bit = offset + _RANK_INDEX[ch]
            if result & (1 << bit):
                raise ValueError(f"Duplicate card {suit}{ch} in {s!r}")
            result |= 1 << bit
    return result


def _spec_to_mask(spec) -> int | None:
    """Return int64 bitmask for str/int specs, None for everything else."""
    if spec is None or callable(spec):
        return None
    if isinstance(spec, int):
        return spec
    if isinstance(spec, str):
        return _parse_partial_hand(spec)
    # HandSet — no fixed cards to extract
    return None


def _mask_to_handset(mask: int):
    """Convert an int64 card-presence bitmask to a HandSet (requires each card)."""
    from .handset import hand_makers
    m = hand_makers
    hs = m.ALL_HANDS
    for bit in range(52):
        if (mask >> bit) & 1:
            suit = _SUITS_BY_OFFSET[bit // 13]
            rank = RANKS[bit % 13]
            hs = hs & m.HAS(f"{suit}{rank}")
    return hs


def random_deals(
    n: int,
    west=None,
    north=None,
    east=None,
    south=None,
    accept=None,
    seed=None,
    fail_count: int = 100_000,
) -> pd.DataFrame:
    """
    Generate *n* random bridge deals, returned as a DataFrame.

    Parameters
    ----------
    n : int
        Number of deals to generate.
    west, north, east, south : optional
        Constraint on each direction's hand. Accepted types:

        - ``None`` — no constraint
        - ``str`` — partial hand string ``"AK/Q/-/-"`` (S/H/D/C format, known cards)
        - ``Hand`` — fix that seat to an exact hand
        - ``int`` — int64 bitmask of required cards
        - ``HandSet`` — BDD constraint (enables fast BDD sampling)
        -- example: ``(h.HCP >= 15) & (h.HCP <= 17) & (h.MATCH_SHAPE("any 4333 + any 4432 + any 5332")``
        - ``callable`` — ``f(Hand) -> bool`` (forces slow accept/reject path)
        -- example: ``lambda h: h.hcp >= 15 and h.hcp <= 17 and h.handshape in [(4,3,3,3),(4,4,3,2),(5,3,3,2)]``

        If none of the four directional constraints are ``callable``, then
        fast sampling is used; however note that sometimes converting
        HandSets into DealSets (which happens internally) can have a one-time
        startup cost of a few seconds.  If any of them are ``callable`` then
        we use a sample/reject strategy, which can become slow if your
        criteria match only rare hands.

    accept : callable, optional
        ``f(Deal) -> bool`` applied to each candidate deal as a post-filter.
    seed : int or numpy.random.Generator, optional
        RNG seed for reproducibility.
    fail_count : int
        A safety valve to protect against impossible constraints;
        after this many consecutive failures before the first success, raise
        ``ValueError``.  ``None`` means no limit.
    """
    specs = [west, north, east, south]

    if accept is None and all(x is None for x in specs):
        # Fastest path: pure numpy shuffle, no BDD overhead
        from .hand import random_deals as _numpy_random_deals
        return _numpy_random_deals(n, seed=seed)

    # BDD sampling is useful when at least one direction has a HandSet/str/int
    # constraint, and no direction uses a plain callable.
    has_bdd_spec = any(x is not None and not (callable(x) and not hasattr(x, "contains"))
                       for x in specs)
    has_plain_callable = any(callable(x) and not hasattr(x, "contains") for x in specs)
    can_use_bdd = has_bdd_spec and not has_plain_callable

    if can_use_bdd:
        if accept is None:
            return _fast_random_deals(n, west, north, east, south, seed)
        return _bdd_random_deals_with_accept(n, west, north, east, south, seed, accept, fail_count)
    return _slow_random_deals(n, west, north, east, south, accept, seed, fail_count)


def _fast_random_deals(n, west, north, east, south, seed):
    """BDD sampling path: all specs are None/str/int/HandSet, no accept."""
    from .handset import hand_makers, HandSet
    m = hand_makers

    def to_hs(spec):
        if spec is None:
            return m.ALL_HANDS
        if isinstance(spec, HandSet):
            return spec
        mask = _spec_to_mask(spec)
        if mask is None:
            raise TypeError(
                f"Unsupported spec type {type(spec).__name__!r} in fast path. "
                "Direction specs must be None, str, int, or HandSet."
            )
        return _mask_to_handset(mask)

    ds = m.WEST(to_hs(west)) & m.NORTH(to_hs(north)) & m.EAST(to_hs(east)) & m.SOUTH(to_hs(south))
    return ds.sample_df(n, seed=seed)


def _bdd_random_deals_with_accept(n, west, north, east, south, seed, accept, fail_count):
    """BDD sampling + accept post-filter.

    Samples batches from the BDD-constrained deal space and applies accept()
    to each deal until n passing deals are collected.
    """
    from .handset import hand_makers, HandSet
    m = hand_makers

    def to_hs(spec):
        if spec is None:
            return m.ALL_HANDS
        if isinstance(spec, HandSet):
            return spec
        mask = _spec_to_mask(spec)
        if mask is None:
            raise TypeError(
                f"Unsupported spec type {type(spec).__name__!r} in fast path. "
                "Direction specs must be None, str, int, or HandSet."
            )
        return _mask_to_handset(mask)

    ds = m.WEST(to_hs(west)) & m.NORTH(to_hs(north)) & m.EAST(to_hs(east)) & m.SOUTH(to_hs(south))

    rng = np.random.default_rng(seed)
    names = ["west", "north", "east", "south"]
    collected: dict[str, list[int]] = {name: [] for name in names}
    hits = 0
    attempts = 0
    batch_size = max(n, 200)

    while hits < n:
        batch_seed = int(rng.integers(0, 2**31))
        batch = ds.sample_df(batch_size, seed=batch_seed)
        for idx in range(len(batch)):
            row = batch.iloc[idx]
            deal = Deal(row)
            if accept(deal):
                for name in names:
                    collected[name].append(int(row[name]))
                hits += 1
                if hits == n:
                    break
        attempts += batch_size
        if fail_count is not None and attempts >= fail_count and hits == 0:
            raise ValueError(
                f"No deals found after {fail_count} BDD-sampled candidates — "
                "is your accept() constraint satisfiable?"
            )

    return pd.DataFrame({
        name: BridgeHandArray(np.array(collected[name], dtype=np.int64))
        for name in names
    })


def _slow_random_deals(n, west, north, east, south, accept, seed, fail_count):
    """Accept/reject path: handles callable specs and/or an accept function."""
    from .handset import HandSet

    rng = np.random.default_rng(seed)
    specs = [west, north, east, south]
    names = ["west", "north", "east", "south"]

    known = [0] * 4       # int bitmask of fixed cards per direction
    acceptors: dict[int, object] = {}  # index → callable(hand_int) → bool
    used_bits: set[int] = set()

    for i, spec in enumerate(specs):
        if spec is None:
            continue
        if isinstance(spec, str):
            mask = _parse_partial_hand(spec)
            _claim_bits(i, mask, known, used_bits)
        elif isinstance(spec, int):
            _claim_bits(i, spec, known, used_bits)
        elif isinstance(spec, HandSet):
            acceptors[i] = spec.contains
        elif callable(spec):
            acceptors[i] = lambda h, f=spec: f(Hand(h))
        else:
            raise TypeError(f"Unsupported spec type: {type(spec)}")

    remaining = np.array([b for b in range(52) if b not in used_bits], dtype=np.intp)
    need = [13 - bin(known[i]).count("1") for i in range(4)]

    if sum(need) != len(remaining):
        raise ValueError(
            f"Impossible deal: known cards total {52 - len(remaining)}, "
            f"but directions need {[13 - n for n in need]} more cards each"
        )

    results: dict[str, list[int]] = {name: [] for name in names}
    hits = 0
    misses = 0

    while hits < n:
        perm = rng.permutation(remaining)
        hands = []
        pos = 0
        for i in range(4):
            k = need[i]
            mask = known[i]
            for b in perm[pos: pos + k]:
                mask |= 1 << int(b)
            hands.append(mask)
            pos += k

        ok = all(acc(hands[i]) for i, acc in acceptors.items())
        if not ok:
            if hits == 0:
                misses += 1
                if fail_count is not None and misses >= fail_count:
                    raise ValueError(
                        f"No deals found after {fail_count} attempts — "
                        "is your constraint satisfiable?"
                    )
            continue

        if accept is not None:
            deal = Deal(west=hands[0], north=hands[1], east=hands[2], south=hands[3])
            if not accept(deal):
                if hits == 0:
                    misses += 1
                    if fail_count is not None and misses >= fail_count:
                        raise ValueError(
                            f"No deals found after {fail_count} attempts — "
                            "is your constraint satisfiable?"
                        )
                continue

        for j, name in enumerate(names):
            results[name].append(hands[j])
        hits += 1

    return pd.DataFrame({
        name: BridgeHandArray(np.array(results[name], dtype=np.int64))
        for name in names
    })


def _claim_bits(idx: int, mask: int, known: list, used_bits: set) -> None:
    bits = {b for b in range(52) if (mask >> b) & 1}
    overlap = used_bits & bits
    if overlap:
        raise ValueError(f"Card(s) assigned to multiple directions: bits {overlap}")
    used_bits |= bits
    known[idx] = mask
