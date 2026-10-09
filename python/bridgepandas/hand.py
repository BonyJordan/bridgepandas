"""
Bridge hand column for pandas.

Each hand is stored as a 52-bit int64 where bit (suit_offset + rank_index) is
set for each held card.  Suit offsets: C=0, D=13, H=26, S=39.
Rank indices: 2=0, 3=1, … 9=7, T=8, J=9, Q=10, K=11, A=12.

Display / parse format is S/H/D/C order, high cards first, e.g. "QJ6/K652/J85/T98".
An empty suit is written as "-".
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pandas.api.extensions import ExtensionArray, ExtensionDtype, register_extension_dtype
from .shape import parse_shape_spec

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANKS = "23456789TJQKA"
_RANK_INDEX: dict[str, int] = {r: i for i, r in enumerate(RANKS)}

# Suit offsets in the bit field
_SUIT_OFFSET: dict[str, int] = {"C": 0, "D": 13, "H": 26, "S": 39}

# Display order: Spades / Hearts / Diamonds / Clubs
_DISPLAY_SUITS = "SHDC"

# Rank bit positions within a 13-bit suit field
_ACE_BIT   = 1 << 12
_KING_BIT  = 1 << 11
_QUEEN_BIT = 1 << 10
_JACK_BIT  = 1 << 9
_TEN_BIT   = 1 << 8


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def hand_str_to_int(s: str) -> int:
    """Parse a hand string like "QJ6/K652/J85/T98" (S/H/D/C) to a 52-bit int."""
    parts = s.split("/")
    if len(parts) != 4:
        raise ValueError(
            f"Hand must have 4 suit groups separated by '/', got: {s!r}"
        )
    value = 0
    for suit, group in zip(_DISPLAY_SUITS, parts):
        if group == "-":
            continue
        offset = _SUIT_OFFSET[suit]
        for ch in group:
            if ch not in _RANK_INDEX:
                raise ValueError(f"Unknown rank {ch!r} in hand {s!r}")
            bit = 1 << (offset + _RANK_INDEX[ch])
            if value & bit:
                raise ValueError(f"Duplicate card {ch}{suit} in hand {s!r}")
            value |= bit
    return value


def int_to_hand_str(value: int) -> str:
    """Convert a 52-bit int to "QJ6/K652/J85/T98" (S/H/D/C) format."""
    parts: list[str] = []
    for suit in _DISPLAY_SUITS:
        offset = _SUIT_OFFSET[suit]
        holding = "".join(
            RANKS[i]
            for i in range(12, -1, -1)  # A down to 2
            if value & (1 << (offset + i))
        )
        parts.append(holding or "-")
    return "/".join(parts)


def _parse_card(card: str) -> int:
    """Parse a suit+rank string like 'CA' or 'SK' into a bit index (0-51)."""
    if len(card) != 2:
        raise ValueError(f"Card must be two characters (suit then rank), got {card!r}")
    suit, rank = card[0].upper(), card[1].upper()
    if suit not in _SUIT_OFFSET:
        raise ValueError(f"Unknown suit {card[0]!r} in {card!r}; use S, H, D, or C")
    if rank not in _RANK_INDEX:
        raise ValueError(f"Unknown rank {card[1]!r} in {card!r}; use A, K, Q, J, T, 9-2")
    return _SUIT_OFFSET[suit] + _RANK_INDEX[rank]


class Hand(int):
    """A single bridge hand stored as a 52-bit integer.

    Subclasses int so it is accepted anywhere a raw hand integer is expected
    (HandSet.contains, numpy arrays, bitwise ops, etc.).  str() and repr()
    display the S/H/D/C hand string instead of the raw number.
    """

    def __new__(cls, value):
        if isinstance(value, str):
            value = hand_str_to_int(value)
        return super().__new__(cls, int(value))

    def __str__(self) -> str:
        return int_to_hand_str(self)

    def __repr__(self) -> str:
        return f"Hand({int_to_hand_str(self)!r})"

    def __len__(self) -> int:
        return int(self).bit_count()

    def __add__(self, card: str) -> Hand:
        return Hand(int(self) | (1 << _parse_card(card)))

    def __sub__(self, card: str) -> Hand:
        return Hand(int(self) & ~(1 << _parse_card(card)))

    def any(self, suit: str) -> bool:
        """Return True if there is at least one card in *suit* ('S', 'H', 'D', or 'C')."""
        suit = suit.upper()
        if suit not in _SUIT_OFFSET:
            raise ValueError(f"Unknown suit {suit!r}; use S, H, D, or C")
        return bool(int(self) & (0x1FFF << _SUIT_OFFSET[suit]))

    # ------------------------------------------------------------------
    # Scalar equivalents of the pandas series accessors
    # ------------------------------------------------------------------

    def _suit_len(self, offset: int) -> int:
        return int(_POPCOUNT13[(int(self) >> offset) & 0x1FFF])

    @property
    def hcp(self) -> int:
        """High card points: A=4, K=3, Q=2, J=1."""
        v = int(self)
        return sum(
            pts
            for offset in (0, 13, 26, 39)
            for rank_idx, pts in ((12, 4), (11, 3), (10, 2), (9, 1))
            if v & (1 << (offset + rank_idx))
        )

    @property
    def akq_points(self) -> int:
        """AKQ points: A=3, K=2, Q=1 (jacks excluded)."""
        v = int(self)
        return sum(
            pts
            for offset in (0, 13, 26, 39)
            for rank_idx, pts in ((12, 3), (11, 2), (10, 1))
            if v & (1 << (offset + rank_idx))
        )

    @property
    def controls(self) -> int:
        """Control count: A=2, K=1."""
        v = int(self)
        return sum(
            pts
            for offset in (0, 13, 26, 39)
            for rank_idx, pts in ((12, 2), (11, 1))
            if v & (1 << (offset + rank_idx))
        )

    @property
    def spades(self) -> int:
        """Number of spades held."""
        return self._suit_len(39)

    @property
    def hearts(self) -> int:
        """Number of hearts held."""
        return self._suit_len(26)

    @property
    def diamonds(self) -> int:
        """Number of diamonds held."""
        return self._suit_len(13)

    @property
    def clubs(self) -> int:
        """Number of clubs held."""
        return self._suit_len(0)

    @property
    def pattern(self) -> tuple[int, int, int, int]:
        """Suit lengths in (spades, hearts, diamonds, clubs) order."""
        v = int(self)
        return (
            int(_POPCOUNT13[(v >> 39) & 0x1FFF]),
            int(_POPCOUNT13[(v >> 26) & 0x1FFF]),
            int(_POPCOUNT13[(v >> 13) & 0x1FFF]),
            int(_POPCOUNT13[v         & 0x1FFF]),
        )

    @property
    def handshape(self) -> tuple[int, int, int, int]:
        """Suit lengths sorted descending, e.g. (5, 4, 3, 1)."""
        return tuple(sorted(self.pattern, reverse=True))

    @property
    def longest_suit(self) -> int:
        """Length of the longest suit."""
        return self.handshape[0]

    @property
    def second_suit(self) -> int:
        """Length of the second-longest suit."""
        return self.handshape[1]

    @property
    def shortest_suit(self) -> int:
        """Length of the shortest suit."""
        return self.handshape[3]

    @property
    def voids(self) -> int:
        """Number of suits in which no card is held."""
        return sum(1 for off in (0, 13, 26, 39) if self._suit_len(off) == 0)

    @property
    def singletons(self) -> int:
        """Number of suits in which exactly one card is held."""
        return sum(1 for off in (0, 13, 26, 39) if self._suit_len(off) == 1)

    @property
    def doubletons(self) -> int:
        """Number of suits in which exactly two cards are held."""
        return sum(1 for off in (0, 13, 26, 39) if self._suit_len(off) == 2)

    def num(self, spec: str) -> int:
        """Count number cards matching a definition.  Some examples:
"A":    number of aces
"S":    number of spades
"HQ":   specifically the queen of hearts
"HDK":  red kings
"A,SK": aces or the king of spades (key cards for spades)
        """
        return (int(self) & _parse_count_spec(spec)).bit_count()

    def suits_of(self, spec: str) -> int:
        """Count suits exactly matching a holding pattern, some examples:
"K":    singleton king
"QJ":   queen-jack tight
"Jx":   jack and exactly one more card lower than the jack
        """
        patterns = _parse_holding_spec(spec)
        v = int(self)
        count = 0
        for offset in (0, 13, 26, 39):
            suit = (v >> offset) & 0x1FFF
            suit_len = int(_POPCOUNT13[suit])
            for total_len, above_mask, req_mask in patterns:
                if suit_len == total_len and (suit & above_mask) == req_mask:
                    count += 1
                    break
        return count

    @property
    def quick_tricks(self) -> float:
        """Quick tricks: AK=2, AQ=1.5, A=1, KQ=1, Kx=0.5."""
        v = int(self)
        total = 0
        for offset in (0, 13, 26, 39):
            suit = (v >> offset) & 0x1FFF
            has_a = bool(suit & _ACE_BIT)
            has_k = bool(suit & _KING_BIT)
            has_q = bool(suit & _QUEEN_BIT)
            has_x = bool(suit & (_JACK_BIT - 1))
            if has_a and has_k:           total += 4
            elif has_a and has_q:         total += 3
            elif has_a:                   total += 2
            elif has_k and has_q:         total += 2
            elif has_k and has_x:         total += 1
        return total * 0.5

    @property
    def losers(self) -> int:
        """Classic losing trick count"""
        v = int(self)
        total = 0
        for offset in (0, 13, 26, 39):
            suit = (v >> offset) & 0x1FFF
            length = int(_POPCOUNT13[suit])
            losers = min(3, length)
            if suit & _ACE_BIT:                      losers -= 1
            if suit & _KING_BIT  and length > 1:   losers -= 1
            if suit & _QUEEN_BIT and length > 2:   losers -= 1
            total += losers
        return total

    def has(self, card: str) -> bool:
        """Return True if this hand contains the given card, e.g. ``"SA"`` for ace of spades."""
        return bool(int(self) & (1 << _parse_card(card)))

    def good_suit(self, spec: str, suit: str) -> bool:
        """Return True if the holding in *suit* matches at least one pattern in the comma-separated *spec*.

        Each pattern is a sequence of ranks followed by optional ``x`` wildcards.
        The holding must be at least as long as the pattern, and each named rank
        must be met or exceeded in that position.  An ``x`` ends the check
        (remaining cards may be anything).

        Examples::

            hand.good_suit("A,Kx", "S")      # spade stopper (A or Kx)
            hand.good_suit("AJx,KQx", "H")   # decent heart suit
        """
        offset = _SUIT_OFFSET[suit.upper()]
        v = int(self)
        ranks_desc = [i for i in range(12, -1, -1) if v & (1 << (offset + i))]
        for pattern in spec.split(","):
            if _match_good_suit(ranks_desc, pattern.strip().upper()):
                return True
        return False

    def match_shape(self, spec: str) -> bool:
        """Return True if this hand's suit-length distribution matches the shape spec.

        Uses the same syntax as ``h.MATCH_SHAPE()``.

        Examples::

            hand.match_shape("any 5332")          # any 5-3-3-2 distribution
            hand.match_shape("4432 + 4333")        # either shape
            hand.match_shape("44xx - 4450")        # 4-4 majors, but not 4=4=5=0
        """
        return self.pattern in parse_shape_spec(spec)


def _match_good_suit(ranks_desc: list, pattern: str) -> bool:
    """Return True if rank list (descending indices) satisfies pattern."""
    if len(ranks_desc) < len(pattern):
        return False
    for pos, p in enumerate(pattern):
        if p == 'X':
            return True
        if ranks_desc[pos] < _RANK_INDEX[p]:
            return False
    return True


def _good_suit_array(data: np.ndarray, patterns: list, suit_offset: int) -> np.ndarray:
    """Return bool array: True if the suit satisfies any pattern."""
    u = data.view(np.uint64)
    suit_bits = ((u >> np.uint64(suit_offset)) & np.uint64(0x1FFF)).astype(np.uint16)
    suit_len = _POPCOUNT13[suit_bits]
    result = np.zeros(len(data), dtype=bool)
    for pattern in patterns:
        match = suit_len >= np.int8(len(pattern))
        for pos, p in enumerate(pattern):
            if not match.any():
                break
            if p == 'X':
                break
            r = _RANK_INDEX[p]
            above_mask = np.uint16(0x1FFF ^ int((1 << r) - 1))
            above_count = _POPCOUNT13[suit_bits & above_mask]
            match = match & (above_count >= np.int8(pos + 1))
        result |= match
    return result


def _match_shape_array(data: np.ndarray, patterns: frozenset) -> np.ndarray:
    """Return bool array: True if the hand's (S,H,D,C) length tuple is in patterns."""
    u = data.view(np.uint64)
    s_len = _POPCOUNT13[((u >> np.uint64(39)) & np.uint64(0x1FFF)).astype(np.uint16)].astype(np.uint16)
    h_len = _POPCOUNT13[((u >> np.uint64(26)) & np.uint64(0x1FFF)).astype(np.uint16)].astype(np.uint16)
    d_len = _POPCOUNT13[((u >> np.uint64(13)) & np.uint64(0x1FFF)).astype(np.uint16)].astype(np.uint16)
    c_len = _POPCOUNT13[((u >> np.uint64(0))  & np.uint64(0x1FFF)).astype(np.uint16)].astype(np.uint16)
    len_vec = (s_len << 12) | (h_len << 8) | (d_len << 4) | c_len
    pat_vecs = np.array([(s << 12) | (h << 8) | (d << 4) | c for s, h, d, c in patterns], dtype=np.uint16)
    return np.isin(len_vec, pat_vecs)


# ---------------------------------------------------------------------------
# Dtype
# ---------------------------------------------------------------------------

@register_extension_dtype
class BridgeHandDtype(ExtensionDtype):
    """Pandas dtype for a bridge hand stored as a 64-bit integer."""

    name = "BridgeHand"
    kind = "i"
    itemsize = 8
    type = int
    na_value = pd.NA

    @classmethod
    def construct_array_type(cls) -> type[BridgeHandArray]:
        return BridgeHandArray


# ---------------------------------------------------------------------------
# Array
# ---------------------------------------------------------------------------

class BridgeHandArray(ExtensionArray):
    """
    Pandas ExtensionArray storing bridge hands as int64 bit fields.

    Construct via pd.array([...], dtype="BridgeHand") or by passing strings
    in S/H/D/C slash-delimited format, raw int64 values, or pd.NA / None.
    """

    dtype = BridgeHandDtype()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, data: np.ndarray, mask: np.ndarray | None = None) -> None:
        if not (isinstance(data, np.ndarray) and data.dtype == np.int64):
            raise TypeError("data must be a numpy int64 array")
        self._data = data
        self._mask = (
            mask
            if mask is not None
            else np.zeros(len(data), dtype=bool)
        )

    @classmethod
    def _from_sequence(
        cls,
        scalars,
        *,
        dtype: BridgeHandDtype | None = None,
        copy: bool = False,
    ) -> BridgeHandArray:
        scalars = list(scalars)
        n = len(scalars)
        data = np.zeros(n, dtype=np.int64)
        mask = np.zeros(n, dtype=bool)
        for i, s in enumerate(scalars):
            if s is None or s is pd.NA:
                mask[i] = True
            elif isinstance(s, str):
                data[i] = hand_str_to_int(s)
            elif isinstance(s, (int, np.integer)):
                data[i] = int(s)
            else:
                raise TypeError(
                    f"Cannot convert {type(s).__name__!r} to BridgeHand; "
                    "expected str (S/H/D/C format), int, or NA"
                )
        return cls(data, mask)

    @classmethod
    def _from_sequence_of_strings(
        cls,
        strings,
        *,
        dtype: BridgeHandDtype | None = None,
        copy: bool = False,
    ) -> BridgeHandArray:
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original: BridgeHandArray) -> BridgeHandArray:
        return cls._from_sequence(values)

    # ------------------------------------------------------------------
    # Element access
    # ------------------------------------------------------------------

    def __getitem__(self, item):
        if isinstance(item, int):
            if self._mask[item]:
                return pd.NA
            return Hand(int(self._data[item]))
        return type(self)(self._data[item].copy(), self._mask[item].copy())

    def __setitem__(self, key, value) -> None:
        if isinstance(value, type(self)):
            self._data[key] = value._data
            self._mask[key] = value._mask
        elif value is None or value is pd.NA:
            self._mask[key] = True
        elif isinstance(value, str):
            self._data[key] = hand_str_to_int(value)
            self._mask[key] = False
        elif isinstance(value, (int, np.integer)):
            self._data[key] = int(value)
            self._mask[key] = False
        else:
            raise TypeError(f"Cannot set BridgeHand from {type(value).__name__!r}")

    def __len__(self) -> int:
        return len(self._data)

    # ------------------------------------------------------------------
    # NA
    # ------------------------------------------------------------------

    def isna(self) -> np.ndarray:
        return self._mask.copy()

    # ------------------------------------------------------------------
    # take (required for slicing, sorting, groupby, …)
    # ------------------------------------------------------------------

    def take(
        self,
        indices,
        *,
        allow_fill: bool = False,
        fill_value=None,
    ) -> BridgeHandArray:
        indices = np.asarray(indices, dtype=np.intp)

        if allow_fill:
            if len(indices) and (indices < -1).any():
                raise ValueError(
                    "take with allow_fill=True requires all indices >= -1"
                )
            if fill_value is None or fill_value is pd.NA:
                fill_data, fill_mask_val = np.int64(0), True
            elif isinstance(fill_value, str):
                fill_data = np.int64(hand_str_to_int(fill_value))
                fill_mask_val = False
            else:
                fill_data, fill_mask_val = np.int64(fill_value), False

            fill_positions = indices == -1
            safe = indices.copy()
            safe[fill_positions] = 0

            result_data = self._data[safe].copy()
            result_mask = self._mask[safe].copy()
            result_data[fill_positions] = fill_data
            result_mask[fill_positions] = fill_mask_val
        else:
            result_data = self._data.take(indices)
            result_mask = self._mask.take(indices)

        return type(self)(result_data, result_mask)

    # ------------------------------------------------------------------
    # copy / concat
    # ------------------------------------------------------------------

    def copy(self) -> BridgeHandArray:
        return type(self)(self._data.copy(), self._mask.copy())

    @classmethod
    def _concat_same_type(cls, to_concat) -> BridgeHandArray:
        return cls(
            np.concatenate([a._data for a in to_concat]),
            np.concatenate([a._mask for a in to_concat]),
        )

    # ------------------------------------------------------------------
    # Factorize (groupby, unique, value_counts)
    # ------------------------------------------------------------------

    def _values_for_factorize(self):
        arr = self._data.copy()
        arr[self._mask] = np.int64(-1)  # -1 is never a valid hand (all bits set)
        return arr, np.int64(-1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def nbytes(self) -> int:
        return self._data.nbytes + self._mask.nbytes

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def _formatter(self, boxed: bool = False):
        def fmt(x) -> str:
            if x is pd.NA:
                return "<NA>"
            return int_to_hand_str(x)
        return fmt

    def __repr__(self) -> str:
        strs = [
            int_to_hand_str(v) if not m else "<NA>"
            for v, m in zip(self._data, self._mask)
        ]
        return f"BridgeHandArray({strs}, dtype={self.dtype!r})"

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def to_strings(self) -> list[str | type(pd.NA)]:
        """Return hand strings (or pd.NA) for each element."""
        return [
            int_to_hand_str(v) if not m else pd.NA
            for v, m in zip(self._data, self._mask)
        ]


# ---------------------------------------------------------------------------
# HCP computation
# ---------------------------------------------------------------------------

# rank_index -> HCP value (only honors matter)
_HCP_BY_RANK: dict[int, int] = {12: 4, 11: 3, 10: 2, 9: 1}  # A, K, Q, J
_SUIT_OFFSETS = (0, 13, 26, 39)


def _hcp_array(data: np.ndarray) -> np.ndarray:
    """Return an int8 array of HCP counts for each int64 hand value."""
    u = data.view(np.uint64)
    hcp = np.zeros(len(u), dtype=np.int8)
    for rank_idx, pts in _HCP_BY_RANK.items():
        for offset in _SUIT_OFFSETS:
            hcp += (((u >> np.uint64(offset + rank_idx)) & np.uint64(1)) * pts).astype(np.int8)
    return hcp


@pd.api.extensions.register_series_accessor("hcp")
class HcpAccessor:
    """Accessor that returns HCP as a real pd.Series via the __new__ trick.

    Because __new__ returns a pd.Series (not an HcpAccessor instance),
    pandas caches and exposes that Series directly.  This means
    ``series.hcp`` IS a Series, so all operators and methods work:

        df[df.west.hcp >= 10]
        df.west.hcp + df.east.hcp
        df.west.hcp.mean()
    """

    def __new__(cls, series: pd.Series) -> pd.Series:
        if not isinstance(series.array, BridgeHandArray):
            raise AttributeError("hcp accessor is only valid for BridgeHand columns")
        arr = series.array
        values = pd.array(_hcp_array(arr._data), dtype=pd.Int8Dtype())
        if arr._mask.any():
            values[arr._mask] = pd.NA
        return pd.Series(values, index=series.index, name=series.name)

    def __init__(self, series: pd.Series) -> None:
        pass  # never called; __new__ returned a non-instance


_AKQ_BY_RANK: dict[int, int] = {12: 3, 11: 2, 10: 1}  # A, K, Q


def _akq_array(data: np.ndarray) -> np.ndarray:
    """Return an int8 array of AKQ point counts (A=3, K=2, Q=1) for each hand."""
    u = data.view(np.uint64)
    akq = np.zeros(len(u), dtype=np.int8)
    for rank_idx, pts in _AKQ_BY_RANK.items():
        for offset in _SUIT_OFFSETS:
            akq += (((u >> np.uint64(offset + rank_idx)) & np.uint64(1)) * pts).astype(np.int8)
    return akq


@pd.api.extensions.register_series_accessor("akq_points")
class AkqAccessor:
    def __new__(cls, series: pd.Series) -> pd.Series:
        if not isinstance(series.array, BridgeHandArray):
            raise AttributeError("akq_points accessor is only valid for BridgeHand columns")
        arr = series.array
        values = pd.array(_akq_array(arr._data), dtype=pd.Int8Dtype())
        if arr._mask.any():
            values[arr._mask] = pd.NA
        return pd.Series(values, index=series.index, name=series.name)

    def __init__(self, series: pd.Series) -> None:
        pass


# ---------------------------------------------------------------------------
# Suit length computation
# ---------------------------------------------------------------------------

# Lookup table: index is a 13-bit value (0–8191), value is its popcount.
_POPCOUNT13 = np.array([bin(i).count("1") for i in range(8192)], dtype=np.int8)


def _suit_length_array(data: np.ndarray, suit_offset: int) -> np.ndarray:
    """Return an int8 array of card counts for one suit across all hands."""
    u = data.view(np.uint64)
    masked = ((u >> np.uint64(suit_offset)) & np.uint64(0x1FFF)).astype(np.uint16)
    return _POPCOUNT13[masked]


def _make_suit_accessor(accessor_name: str, suit_offset: int):
    @pd.api.extensions.register_series_accessor(accessor_name)
    class _SuitAccessor:
        def __new__(cls, series: pd.Series) -> pd.Series:
            if not isinstance(series.array, BridgeHandArray):
                raise AttributeError(
                    f"{accessor_name} accessor is only valid for BridgeHand columns"
                )
            arr = series.array
            values = pd.array(
                _suit_length_array(arr._data, suit_offset), dtype=pd.Int8Dtype()
            )
            if arr._mask.any():
                values[arr._mask] = pd.NA
            return pd.Series(values, index=series.index, name=series.name)

        def __init__(self, series: pd.Series) -> None:
            pass

    _SuitAccessor.__name__ = f"{accessor_name.capitalize()}Accessor"
    _SuitAccessor.__qualname__ = f"{accessor_name.capitalize()}Accessor"
    return _SuitAccessor


SpadesAccessor   = _make_suit_accessor("spades",   39)
HeartsAccessor   = _make_suit_accessor("hearts",   26)
DiamondsAccessor = _make_suit_accessor("diamonds", 13)
ClubsAccessor    = _make_suit_accessor("clubs",     0)


# ---------------------------------------------------------------------------
# Shape accessors (voids / singletons / doubletons)
# ---------------------------------------------------------------------------

def _shape_count_array(data: np.ndarray, target_length: int) -> np.ndarray:
    """Return an int8 array counting how many suits have exactly target_length cards."""
    u = data.view(np.uint64)
    counts = np.zeros(len(u), dtype=np.int8)
    for offset in _SUIT_OFFSETS:
        suit_bits = ((u >> np.uint64(offset)) & np.uint64(0x1FFF)).astype(np.uint16)
        counts += (_POPCOUNT13[suit_bits] == target_length).astype(np.int8)
    return counts


def _make_shape_accessor(accessor_name: str, target_length: int):
    @pd.api.extensions.register_series_accessor(accessor_name)
    class _ShapeAccessor:
        def __new__(cls, series: pd.Series) -> pd.Series:
            if not isinstance(series.array, BridgeHandArray):
                raise AttributeError(
                    f"{accessor_name} accessor is only valid for BridgeHand columns"
                )
            arr = series.array
            values = pd.array(
                _shape_count_array(arr._data, target_length), dtype=pd.Int8Dtype()
            )
            if arr._mask.any():
                values[arr._mask] = pd.NA
            return pd.Series(values, index=series.index, name=series.name)

        def __init__(self, series: pd.Series) -> None:
            pass

    _ShapeAccessor.__name__ = f"{accessor_name.capitalize()}Accessor"
    _ShapeAccessor.__qualname__ = f"{accessor_name.capitalize()}Accessor"
    return _ShapeAccessor


VoidsAccessor      = _make_shape_accessor("voids",      0)
SingletonsAccessor = _make_shape_accessor("singletons", 1)
DoubletonsAccessor = _make_shape_accessor("doubletons", 2)


# ---------------------------------------------------------------------------
# Sorted suit length accessors (longest_suit, second_suit, shortest_suit)
# ---------------------------------------------------------------------------

def _sorted_suit_lengths(data: np.ndarray) -> np.ndarray:
    """Return (n, 4) int8 array of suit lengths sorted descending."""
    lengths = np.stack(
        [_suit_length_array(data, off) for off in (0, 13, 26, 39)], axis=1
    )
    return np.sort(lengths, axis=1)[:, ::-1]


def _make_sorted_length_accessor(accessor_name: str, place: int):
    @pd.api.extensions.register_series_accessor(accessor_name)
    class _SortedLenAccessor:
        def __new__(cls, series: pd.Series) -> pd.Series:
            if not isinstance(series.array, BridgeHandArray):
                raise AttributeError(
                    f"{accessor_name} accessor is only valid for BridgeHand columns"
                )
            arr = series.array
            values = pd.array(
                _sorted_suit_lengths(arr._data)[:, place], dtype=pd.Int8Dtype()
            )
            if arr._mask.any():
                values[arr._mask] = pd.NA
            return pd.Series(values, index=series.index, name=series.name)

        def __init__(self, series: pd.Series) -> None:
            pass

    _SortedLenAccessor.__name__ = f"{accessor_name.capitalize()}Accessor"
    _SortedLenAccessor.__qualname__ = f"{accessor_name.capitalize()}Accessor"
    return _SortedLenAccessor


LongestSuitAccessor = _make_sorted_length_accessor("longest_suit", 0)
SecondSuitAccessor  = _make_sorted_length_accessor("second_suit",  1)
ShortestSuitAccessor = _make_sorted_length_accessor("shortest_suit", 3)


@pd.api.extensions.register_series_accessor("handshape")
class HandshapeAccessor:
    """Series accessor returning a Series of suit-length tuples sorted descending."""

    def __new__(cls, series: pd.Series) -> pd.Series:
        if not isinstance(series.array, BridgeHandArray):
            raise AttributeError("handshape accessor is only valid for BridgeHand columns")
        arr = series.array
        lengths = _sorted_suit_lengths(arr._data)
        tuples = [tuple(row) for row in lengths.tolist()]
        result = pd.Series(tuples, index=series.index, name=series.name, dtype=object)
        if arr._mask.any():
            result[arr._mask] = None
        return result

    def __init__(self, series: pd.Series) -> None:
        pass


# ---------------------------------------------------------------------------
# Exact suit holding accessor  (series.suits("Kx") etc.)
# ---------------------------------------------------------------------------

# Token: zero or more rank chars followed by zero or more 'x' wildcards.
# Each 'x' means one additional card strictly below the lowest named rank.
_HOLDING_TOKEN_RE = re.compile(r"^([AKQJTakqjt2-9]*)([xX]*)$")


def _parse_holding_token(token: str) -> tuple[int, int, int]:
    """Parse one holding pattern into (total_length, above_mask, required_mask).

    above_mask is a 13-bit mask covering all ranks >= the lowest named rank.
    The match condition is:
        popcount(suit) == total_length  AND  (suit & above_mask) == required_mask
    """
    mo = _HOLDING_TOKEN_RE.match(token)
    if not mo or not (mo.group(1) or mo.group(2)):
        raise ValueError(
            f"Invalid holding token {token!r}. "
            "Use ranks (AKQJT2-9) then optional x wildcards, e.g. 'K', 'Qx', 'AKJ'."
        )
    ranks_str = mo.group(1).upper()
    x_count = len(mo.group(2))

    if len(set(ranks_str)) != len(ranks_str):
        raise ValueError(f"Duplicate rank in holding token {token!r}")

    named_indices = [_RANK_INDEX[r] for r in ranks_str]
    total_length = len(named_indices) + x_count

    if not named_indices:
        return (total_length, 0, 0)  # any suit of that length

    req_mask = 0
    for i in named_indices:
        req_mask |= 1 << i
    lowest = min(named_indices)
    above_mask = (1 << 13) - (1 << lowest)  # bits lowest..12 inclusive

    return (total_length, above_mask, req_mask)


def _parse_holding_spec(spec: str) -> list[tuple[int, int, int]]:
    """Parse a comma-separated holding spec into a list of patterns."""
    patterns = [
        _parse_holding_token(t.strip())
        for t in spec.split(",")
        if t.strip()
    ]
    if not patterns:
        raise ValueError(f"Empty holding spec: {spec!r}")
    return patterns


def _suits_array(data: np.ndarray, patterns: list[tuple[int, int, int]]) -> np.ndarray:
    """Return int8 array: count of suits matching any holding pattern."""
    u = data.view(np.uint64)
    counts = np.zeros(len(u), dtype=np.int8)
    for offset in _SUIT_OFFSETS:
        suit = ((u >> np.uint64(offset)) & np.uint64(0x1FFF)).astype(np.uint16)
        suit_len = _POPCOUNT13[suit]
        matches = np.zeros(len(u), dtype=bool)
        for total_len, above_mask, req_mask in patterns:
            length_ok = suit_len == total_len
            honor_ok = (suit & np.uint16(above_mask)) == np.uint16(req_mask)
            matches |= length_ok & honor_ok
        counts += matches.astype(np.int8)
    return counts


@pd.api.extensions.register_series_accessor("suits_of")
class _SuitsOfAccessor:
    def __new__(cls, series: pd.Series):
        arr = series.array
        if not isinstance(arr, BridgeHandArray):
            raise AttributeError("suits_of is only valid for BridgeHand series")

        def _suits_of(spec: str) -> pd.Series:
            patterns = _parse_holding_spec(spec)
            counts = _suits_array(arr._data, patterns)
            values = pd.array(counts, dtype=pd.Int8Dtype())
            if arr._mask.any():
                values[arr._mask] = pd.NA
            return pd.Series(values, index=series.index, name=series.name)

        return _suits_of

    def __init__(self, series: pd.Series) -> None:
        pass


# ---------------------------------------------------------------------------
# Single-card membership  (series.has("SK"))
# ---------------------------------------------------------------------------

@pd.api.extensions.register_series_accessor("has")
class _HasAccessor:
    def __new__(cls, series: pd.Series):
        arr = series.array
        if not isinstance(arr, BridgeHandArray):
            raise AttributeError("has is only valid for BridgeHand series")

        def _has(card: str) -> pd.Series:
            bit = _parse_card(card)
            present = (arr._data & np.int64(1 << bit)) != 0
            values = pd.array(present, dtype=pd.BooleanDtype())
            if arr._mask.any():
                values[arr._mask] = pd.NA
            return pd.Series(values, index=series.index, name=series.name)

        return _has

    def __init__(self, series: pd.Series) -> None:
        pass


# ---------------------------------------------------------------------------
# Generic card counting
# ---------------------------------------------------------------------------

# Each token in the comma-separated spec is suits (SHDC) then ranks (AKQJT2-9).
# Either part may be omitted; omitting suits means all suits, omitting ranks
# means all ranks in those suits.  E.g. "HDAK" = red A/K; "A,SK" = all aces
# plus SK (key cards for spades).
_COUNT_TOKEN_RE = re.compile(r"^([SHDCshdc]*)([AKQJTakqjt2-9]*)$")


def _parse_count_spec(spec: str) -> int:
    """Parse a count spec string into a 52-bit card bitmask."""
    mask = 0
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        mo = _COUNT_TOKEN_RE.match(token)
        if not mo or (not mo.group(1) and not mo.group(2)):
            raise ValueError(
                f"Bad count spec token {token!r}. "
                "Each token must be suits (SHDC) then ranks (AKQJT2-9), e.g. 'HDA' or 'SK'."
            )
        suits = mo.group(1).upper() or "SHDC"
        ranks = mo.group(2).upper() or RANKS
        for suit in suits:
            off = _SUIT_OFFSET[suit]
            for rank in ranks:
                mask |= 1 << (off + _RANK_INDEX[rank])
    if not mask:
        raise ValueError(f"Empty count spec: {spec!r}")
    return mask


def _count_spec_array(data: np.ndarray, spec_mask: int) -> np.ndarray:
    """Return an int8 array of how many cards matching spec_mask each hand holds."""
    masked = (data & spec_mask).astype(np.uint64).view(np.uint8)
    return np.unpackbits(masked).reshape(-1, 64).sum(axis=1).astype(np.int8)


@pd.api.extensions.register_series_accessor("num")
class _NumAccessor:
    """Accessor that returns a callable via the __new__ trick.

    ``series.num`` evaluates to a function, so
    ``series.num("A,SK")`` returns a pd.Series of counts.
    """

    def __new__(cls, series: pd.Series):
        arr = series.array
        if not isinstance(arr, BridgeHandArray):
            raise AttributeError("num is only valid for BridgeHand series")

        def _count(spec: str) -> pd.Series:
            sm = _parse_count_spec(spec)
            counts = _count_spec_array(arr._data, sm)
            values = pd.array(counts, dtype=pd.Int8Dtype())
            if arr._mask.any():
                values[arr._mask] = pd.NA
            return pd.Series(values, index=series.index, name=series.name)

        return _count

    def __init__(self, series: pd.Series) -> None:
        pass  # never called; __new__ returned a non-instance


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------

# Bitmask of all aces and kings (the only cards that score controls).
# Aces are at offset+12, kings at offset+11, for offsets C=0,D=13,H=26,S=39.
_HONORS_MASK = np.uint64(0x000c006003001800)

def _controls_array(data: np.ndarray) -> np.ndarray:
    """Return an int8 array of control counts (A=2, K=1) for each hand.

    The bit trick sums the four 2-bit (ace, king) pairs in two parallel add
    steps then normalises to bits 0-3:

        a — extract the 8 honor bits
        b — fold S+H and D+C into two 3-bit accumulators at bits 37-39 / 11-13
        c — add the two accumulators and shift to bits 0-3
    """
    u = data.view(np.uint64)
    a = u & _HONORS_MASK
    b = (a + (a >> np.uint64(13))) & np.uint64(0x000e000003800)
    c = ((b + (b >> np.uint64(26))) >> np.uint64(11)) & np.uint64(0xf)
    return c.astype(np.int8)


@pd.api.extensions.register_series_accessor("controls")
class ControlsAccessor:
    def __new__(cls, series: pd.Series) -> pd.Series:
        if not isinstance(series.array, BridgeHandArray):
            raise AttributeError("controls is only valid for BridgeHand series")
        arr = series.array
        values = pd.array(_controls_array(arr._data), dtype=pd.Int8Dtype())
        if arr._mask.any():
            values[arr._mask] = pd.NA
        return pd.Series(values, index=series.index, name=series.name)

    def __init__(self, series: pd.Series) -> None:
        pass


# ---------------------------------------------------------------------------
# Total card count
# ---------------------------------------------------------------------------

@pd.api.extensions.register_series_accessor("length")
class LengthAccessor:
    def __new__(cls, series: pd.Series) -> pd.Series:
        if not isinstance(series.array, BridgeHandArray):
            raise AttributeError("length accessor is only valid for BridgeHand columns")
        arr = series.array
        u = arr._data.view(np.uint64)
        counts = sum(
            _POPCOUNT13[((u >> np.uint64(off)) & np.uint64(0x1FFF)).astype(np.uint16)]
            for off in _SUIT_OFFSETS
        ).astype(np.int8)
        values = pd.array(counts, dtype=pd.Int8Dtype())
        if arr._mask.any():
            values[arr._mask] = pd.NA
        return pd.Series(values, index=series.index, name=series.name)

    def __init__(self, series: pd.Series) -> None:
        pass


# ---------------------------------------------------------------------------
# Losing trick count
# ---------------------------------------------------------------------------

def _losers_array(data: np.ndarray) -> np.ndarray:
    """Return an int8 array of losing trick counts for each hand."""
    u = data.view(np.uint64)
    total = np.zeros(len(u), dtype=np.int8)
    for offset in _SUIT_OFFSETS:
        suit = ((u >> np.uint64(offset)) & np.uint64(0x1FFF)).astype(np.uint16)
        length = _POPCOUNT13[suit]
        losers = np.minimum(length, np.int8(3))
        losers -= ((suit & np.uint16(_ACE_BIT))   != 0).astype(np.int8)
        losers -= (((suit & np.uint16(_KING_BIT))  != 0) & (length > 1)).astype(np.int8)
        losers -= (((suit & np.uint16(_QUEEN_BIT)) != 0) & (length > 2)).astype(np.int8)
        total += losers
    return total


@pd.api.extensions.register_series_accessor("losers")
class LosersAccessor:
    def __new__(cls, series: pd.Series) -> pd.Series:
        if not isinstance(series.array, BridgeHandArray):
            raise AttributeError("losers accessor is only valid for BridgeHand columns")
        arr = series.array
        values = pd.array(_losers_array(arr._data), dtype=pd.Int8Dtype())
        if arr._mask.any():
            values[arr._mask] = pd.NA
        return pd.Series(values, index=series.index, name=series.name)

    def __init__(self, series: pd.Series) -> None:
        pass


# ---------------------------------------------------------------------------
# Quick tricks
# ---------------------------------------------------------------------------


def _quick_tricks_array(data: np.ndarray) -> np.ndarray:
    """Return a float32 array of quick trick counts (AK=2, AQ=1.5, A=1, KQ=1, Kx=0.5).

    Accumulated as int8 (×2) per suit then multiplied by 0.5 at the end.
    """
    u = data.view(np.uint64)
    total = np.zeros(len(u), dtype=np.int8)
    for offset in _SUIT_OFFSETS:
        suit = ((u >> np.uint64(offset)) & np.uint64(0x1FFF)).astype(np.uint16)
        has_a = (suit & np.uint16(_ACE_BIT))   != 0
        has_k = (suit & np.uint16(_KING_BIT))  != 0
        has_q = (suit & np.uint16(_QUEEN_BIT)) != 0
        has_x = (suit & np.uint16(_JACK_BIT - 1)) != 0   # J through 2
        total += (has_a & has_k).astype(np.int8)             * np.int8(4)
        total += (has_a & ~has_k & has_q).astype(np.int8)    * np.int8(3)
        total += (has_a & ~has_k & ~has_q).astype(np.int8)   * np.int8(2)
        total += (~has_a & has_k & has_q).astype(np.int8)    * np.int8(2)
        total += (~has_a & has_k & ~has_q & has_x).astype(np.int8)
    return total.astype(np.float32) * np.float32(0.5)


@pd.api.extensions.register_series_accessor("quick_tricks")
class QuickTricksAccessor:
    def __new__(cls, series: pd.Series) -> pd.Series:
        if not isinstance(series.array, BridgeHandArray):
            raise AttributeError("quick_tricks accessor is only valid for BridgeHand columns")
        arr = series.array
        values = pd.array(_quick_tricks_array(arr._data), dtype=pd.Float32Dtype())
        if arr._mask.any():
            values[arr._mask] = pd.NA
        return pd.Series(values, index=series.index, name=series.name)

    def __init__(self, series: pd.Series) -> None:
        pass


# ---------------------------------------------------------------------------
# Good-suit accessor
# ---------------------------------------------------------------------------

@pd.api.extensions.register_series_accessor("good_suit")
class _GoodSuitAccessor:
    def __new__(cls, series: pd.Series):
        arr = series.array
        if not isinstance(arr, BridgeHandArray):
            raise AttributeError("good_suit is only valid for BridgeHand series")

        def _good_suit(spec: str, suit: str) -> pd.Series:
            patterns = [p.strip().upper() for p in spec.split(",") if p.strip()]
            offset = _SUIT_OFFSET[suit.upper()]
            result = _good_suit_array(arr._data, patterns, offset)
            values = pd.array(result, dtype=pd.BooleanDtype())
            if arr._mask.any():
                values[arr._mask] = pd.NA
            return pd.Series(values, index=series.index, name=series.name)

        return _good_suit

    def __init__(self, series: pd.Series) -> None:
        pass


@pd.api.extensions.register_series_accessor("match_shape")
class _MatchShapeAccessor:
    def __new__(cls, series: pd.Series):
        arr = series.array
        if not isinstance(arr, BridgeHandArray):
            raise AttributeError("match_shape is only valid for BridgeHand series")

        def _match_shape(spec: str) -> pd.Series:
            patterns = parse_shape_spec(spec)
            result = _match_shape_array(arr._data, patterns)
            values = pd.array(result, dtype=pd.BooleanDtype())
            if arr._mask.any():
                values[arr._mask] = pd.NA
            return pd.Series(values, index=series.index, name=series.name)

        return _match_shape

    def __init__(self, series: pd.Series) -> None:
        pass


# ---------------------------------------------------------------------------
# Deal generation
# ---------------------------------------------------------------------------

_DIRECTIONS = ["west", "north", "east", "south"]


def random_deals(n: int, seed=None) -> pd.DataFrame:
    """Return a DataFrame of n random bridge deals.

    Columns are west, north, east, south, each a BridgeHandArray.
    Each row is a full deal: all 52 cards distributed 13 per hand.

    Parameters
    ----------
    n : int
        Number of deals to generate.
    seed : int or numpy.random.Generator, optional
        Seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    # Build n independent shuffles of the 52-card deck (one per row).
    deck = np.broadcast_to(np.arange(52, dtype=np.int64), (n, 52)).copy()
    perm = rng.permuted(deck, axis=1)  # shape (n, 52)
    # Convert each 13-card slice to a bit field via OR-reduce of (1 << card_index).
    return pd.DataFrame({
        name: BridgeHandArray(
            np.bitwise_or.reduce(np.int64(1) << perm[:, i * 13 : (i + 1) * 13], axis=1)
        )
        for i, name in enumerate(_DIRECTIONS)
    })
