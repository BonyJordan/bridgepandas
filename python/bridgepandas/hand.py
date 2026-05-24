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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANKS = "23456789TJQKA"
_RANK_INDEX: dict[str, int] = {r: i for i, r in enumerate(RANKS)}

# Suit offsets in the bit field
_SUIT_OFFSET: dict[str, int] = {"C": 0, "D": 13, "H": 26, "S": 39}

# Display order: Spades / Hearts / Diamonds / Clubs
_DISPLAY_SUITS = "SHDC"


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
            return int(self._data[item])
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

def _controls_array(data: np.ndarray, singleton_kings: bool) -> np.ndarray:
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
    controls = c.astype(np.int8)
    if not singleton_kings:
        singleton = np.uint64(1 << 11)  # king is the only card in the suit
        for off in (np.uint64(0), np.uint64(13), np.uint64(26), np.uint64(39)):
            suit = (u >> off) & np.uint64(0x1FFF)
            controls -= (suit == singleton).astype(np.int8)
    return controls


@pd.api.extensions.register_series_accessor("controls")
class ControlsAccessor:
    def __new__(cls, series: pd.Series) -> pd.Series:
        if not isinstance(series.array, BridgeHandArray):
            raise AttributeError("controls is only valid for BridgeHand series")
        arr = series.array
        values = pd.array(_controls_array(arr._data, singleton_kings=True), dtype=pd.Int8Dtype())
        if arr._mask.any():
            values[arr._mask] = pd.NA
        return pd.Series(values, index=series.index, name=series.name)

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
