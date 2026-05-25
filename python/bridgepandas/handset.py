"""
Constrained deal sampling via Binary Decision Diagrams.

The main entry point is ``hand_makers``, which provides lazily-computed
metrics (HCP, suit lengths, shape, …) that can be combined with boolean
operators to produce a ``HandSet`` or ``DealSet``.  Both support sampling
and exact counting via the jbdd C extension.

BDD variable ordering for HandSet (52 variables):
  vars  0-15 : honors A/K/Q/J interleaved by suit in SHDC order
               (SA=0, HA=1, DA=2, CA=3, SK=4, HK=5, …)
  vars 16-51 : T through 2, by suit in SHDC order
               (ST=16, HT=17, DT=18, CT=19, S9=20, …)

For DealSet (104 variables, 2 per card):
  vars 2i, 2i+1 encode which player holds card i:
    owner = 2*(bit 2i+1 is set) + 1*(bit 2i is set)
    West=0, North=1, East=2, South=3
"""

from __future__ import annotations

import collections
import functools
import itertools
import operator
import random

import numpy as np
import pandas as pd

from .jbdd import BDD
from .hand import BridgeHandArray, _SUIT_OFFSET, _RANK_INDEX, _parse_count_spec
from .shape import parse_shape_spec, _tuple_to_pattern

# ---------------------------------------------------------------------------
# Card type (suit in SHDC, rank in AKQJT98765432)
# ---------------------------------------------------------------------------

Card = collections.namedtuple("Card", "suit rank")

# ---------------------------------------------------------------------------
# BDD variable ordering
# ---------------------------------------------------------------------------

_BDD_CARDS: list[Card] = (
    [Card(suit, rank) for rank in "AKQJ"        for suit in "SHDC"] +
    [Card(suit, rank) for suit in "SHDC" for rank in "T98765432"]
)
assert len(_BDD_CARDS) == 52

_BDD_VAR: dict[Card, int] = {c: i for i, c in enumerate(_BDD_CARDS)}


# ---------------------------------------------------------------------------
# Conversion helpers: BDD variable sets ↔ int64 hand encoding
# ---------------------------------------------------------------------------

def _card_bit(card: Card) -> int:
    """Bit position of a card in our int64 encoding (identical to bridgemoose bit_pack)."""
    return _SUIT_OFFSET[card.suit] + _RANK_INDEX[card.rank]


def _hand_vars_to_int64(var_indices) -> int:
    """HandSet BDD variable indices → int64 hand."""
    out = 0
    for v in var_indices:
        out |= 1 << _card_bit(_BDD_CARDS[v])
    return out


def _deal_vars_to_int64s(bit_set) -> list[int]:
    """DealSet BDD variable set → [West, North, East, South] int64s."""
    bit_set = set(bit_set)
    hands = [0, 0, 0, 0]
    for i, card in enumerate(_BDD_CARDS):
        owner = 2 * (2*i+1 in bit_set) + (2*i in bit_set)
        hands[owner] |= 1 << _card_bit(card)
    return hands


def _int64s_to_deal_vars(west: int, north: int, east: int, south: int) -> set[int]:
    """[West, North, East, South] int64s → DealSet BDD variable set."""
    bits: set[int] = set()
    deal = [west, north, east, south]
    for i, card in enumerate(_BDD_CARDS):
        mask = 1 << _card_bit(card)
        for owner, hand in enumerate(deal):
            if hand & mask:
                if owner & 1:
                    bits.add(2*i)
                if owner & 2:
                    bits.add(2*i+1)
                break
    return bits


# ---------------------------------------------------------------------------
# HandSetMetric
# ---------------------------------------------------------------------------

class HandSetMetric:
    def __init__(self, values: dict[int, BDD]):
        self.cache: dict = {}
        self.values = values

    def __eq__(self, other):
        if isinstance(other, HandSetMetric):
            return (self - other) == 0
        elif isinstance(other, int):
            key = ("=", other)
            if key not in self.cache:
                self.cache[key] = self.values.get(other, BDD.false())
            return HandSet(self.cache[key])
        raise TypeError(other)

    def __ne__(self, other):
        return ~(self == other)

    def _cmp(op, from_int):
        def fn(self, other):
            if isinstance(other, HandSetMetric):
                return op(self - other, 0)
            elif isinstance(other, int):
                return from_int(self, other)
            raise TypeError(other)
        return fn

    __le__ = _cmp(operator.le, lambda self, n: self.less_than(n + 1))
    __lt__ = _cmp(operator.lt, lambda self, n: self.less_than(n))
    __ge__ = _cmp(operator.ge, lambda self, n: ~self.less_than(n))
    __gt__ = _cmp(operator.gt, lambda self, n: ~self.less_than(n + 1))

    def _arith(op):
        def fn(self, other):
            if not isinstance(other, HandSetMetric):
                raise TypeError(other)
            out: dict[int, BDD] = {}
            for k1, v1 in self.values.items():
                for k2, v2 in other.values.items():
                    k3 = op(k1, k2)
                    both = v1 & v2
                    out[k3] = out[k3] | both if k3 in out else both
            return HandSetMetric(out)
        return fn

    __add__ = _arith(operator.add)
    __sub__ = _arith(operator.sub)

    def __mul__(self, other):
        return HandSetMetric({k * other: v for k, v in self.values.items()})

    __rmul__ = __mul__

    def less_than(self, n: int) -> HandSet:
        if n <= min(self.values):
            return HandSet(BDD.false())
        key = ("<", n)
        if key not in self.cache:
            a = self.less_than(n - 1).bdd
            b = self.values.get(n - 1, BDD.false())
            self.cache[key] = a | b
        return HandSet(self.cache[key])


# ---------------------------------------------------------------------------
# SimpleHandMetric
# ---------------------------------------------------------------------------

class SimpleHandMetric(HandSetMetric):
    """
    Metric built from per-card integer scores.  Scores is a dict mapping
    Card → int (e.g. {Card('S','A'): 4, Card('S','K'): 3, …} for HCP).
    """

    def __init__(self, scores: dict[Card, int]):
        values: dict[int, BDD] = {0: BDD.true()}
        for card in reversed(sorted(scores, key=_BDD_VAR.__getitem__)):
            pts = scores[card]
            var = _BDD_VAR[card]
            avec_vals = {val + pts: bdd for val, bdd in values.items()}
            sans_vals = values
            values = {}
            for val in avec_vals.keys() | sans_vals.keys():
                a = avec_vals.get(val, BDD.false())
                s = sans_vals.get(val, BDD.false())
                values[val] = BDD(var).thenelse(a, s)
        super().__init__(values)


# ---------------------------------------------------------------------------
# QuickTricksMetric
# ---------------------------------------------------------------------------

class QuickTricksMetric(HandSetMetric):
    """
    Counts quick tricks × 2 (integer) per hand, summed across suits.
    AK=4, AQ=3, A=2 or KQ=2, Kx=1, else 0.
    """

    def __init__(self):
        suit_val_lists = [QuickTricksMetric._suit_values(s).items() for s in "SHDC"]
        values: dict[int, BDD] = {}
        for combo in itertools.product(*suit_val_lists):
            key = sum(x[0] for x in combo)
            val = functools.reduce(BDD.__and__, [x[1] for x in combo])
            values[key] = values[key] | val if key in values else val
        super().__init__(values)

    @staticmethod
    def _suit_values(suit: str) -> dict[int, BDD]:
        suit_vars = [v for v, c in enumerate(_BDD_CARDS) if c.suit == suit]
        # suit_vars[0]=A, [1]=K, [2]=Q, [3]=J, [4:]=T-2
        have_a = BDD(suit_vars[0])
        have_k = BDD(suit_vars[1])
        have_q = BDD(suit_vars[2])
        have_x = BDD.false()
        for v in reversed(suit_vars[3:]):
            have_x = BDD(v).thenelse(BDD.true(), have_x)

        vals = {
            4: have_a & have_k,
            3: have_a & ~have_k & have_q,
            2: (have_a & ~have_k & ~have_q) | (~have_a & have_k & have_q),
            1: ~have_a & have_k & ~have_q & have_x,
            0: ~have_a & have_k.thenelse(~have_q & ~have_x, BDD.true()),
        }
        assert functools.reduce(BDD.__or__, vals.values()) == BDD.true()
        return vals


# ---------------------------------------------------------------------------
# HandSet
# ---------------------------------------------------------------------------

class HandSet:
    """A BDD representing a set of 13-card hands for one player."""

    _NUM_CARDS = None  # lazy
    _HAND_BDD  = None  # lazy: BDD for "exactly 13 cards"

    @classmethod
    def _ensure_hand_bdd(cls):
        if cls._HAND_BDD is None:
            cls._NUM_CARDS = SimpleHandMetric({c: 1 for c in _BDD_CARDS})
            cls._HAND_BDD  = cls._NUM_CARDS.values[13]

    def __init__(self, bdd: BDD):
        HandSet._ensure_hand_bdd()
        self.bdd = bdd & HandSet._HAND_BDD

    # --- set operations ---

    def __and__(self, other: HandSet)  -> HandSet: return HandSet(self.bdd & other.bdd)
    def __or__ (self, other: HandSet)  -> HandSet: return HandSet(self.bdd | other.bdd)
    def __xor__(self, other: HandSet)  -> HandSet: return HandSet(self.bdd ^ other.bdd)
    def __invert__(self)               -> HandSet: return HandSet(~self.bdd)
    def ite(self, t: HandSet, e: HandSet) -> HandSet:
        return HandSet(self.bdd.thenelse(t.bdd, e.bdd))

    # --- query ---

    def count(self) -> int:
        return self.bdd.pcount()

    def contains(self, hand_int: int) -> bool:
        """Return True if hand_int (int64) is in this HandSet."""
        ones = {v for v, c in enumerate(_BDD_CARDS) if hand_int & (1 << _card_bit(c))}
        return bool(self.bdd.eval_pset(ones))

    def sample(self, rng=random) -> int:
        """Return a random hand as an int64."""
        idx = rng.randrange(self.bdd.pcount())
        return _hand_vars_to_int64(self.bdd.get_pindex(idx))


# ---------------------------------------------------------------------------
# DealSet
# ---------------------------------------------------------------------------

class DealSet:
    """A BDD representing a set of full deals (all 52 cards distributed 13 each)."""

    def __init__(self, d: BDD):
        self.d = d if d is not None else BDD.false()

    # --- set operations ---

    def __and__(self, other: DealSet)  -> DealSet: return DealSet(self.d & other.d)
    def __or__ (self, other: DealSet)  -> DealSet: return DealSet(self.d | other.d)
    def __xor__(self, other: DealSet)  -> DealSet: return DealSet(self.d ^ other.d)
    def __invert__(self)               -> DealSet: return DealSet(~self.d)
    def ite(self, t: DealSet, e: DealSet) -> DealSet:
        return DealSet(self.d.thenelse(t.d, e.d))

    # --- query ---

    def count(self) -> int:
        return self.d.pcount()

    def contains(self, west: int, north: int, east: int, south: int) -> bool:
        """Return True if the given deal (four int64 hands) is in this DealSet."""
        return bool(self.d.eval_pset(_int64s_to_deal_vars(west, north, east, south)))

    def sample(self, rng=random) -> dict[str, int]:
        """Return one random deal as {west, north, east, south} int64 dict."""
        idx = rng.randrange(self.d.pcount())
        w, n, e, s = _deal_vars_to_int64s(self.d.get_pindex(idx))
        return {"west": w, "north": n, "east": e, "south": s}

    def sample_df(self, n: int, seed=None) -> pd.DataFrame:
        """Return n random deals as a DataFrame with BridgeHandArray columns."""
        rng = random.Random(seed)
        count = self.d.pcount()
        cols: dict[str, list[int]] = {"west": [], "north": [], "east": [], "south": []}
        for _ in range(n):
            idx = rng.randrange(count)
            w, no, e, s = _deal_vars_to_int64s(self.d.get_pindex(idx))
            cols["west"].append(w)
            cols["north"].append(no)
            cols["east"].append(e)
            cols["south"].append(s)
        return pd.DataFrame({
            name: BridgeHandArray(np.array(vals, dtype=np.int64))
            for name, vals in cols.items()
        })


# ---------------------------------------------------------------------------
# DealSetConverter  (HandSet for one player → DealSet)
# ---------------------------------------------------------------------------

_DIRECTION_INDEX = {"W": 0, "N": 1, "E": 2, "S": 3}


class DealSetConverter:
    """Converts a HandSet constraint for one player into a DealSet."""

    _four_hands: BDD | None = None

    def __init__(self, player: str):
        self._pi = _DIRECTION_INDEX[player.upper()]
        self._cache: dict = {}

    def __call__(self, hs: HandSet) -> DealSet:
        if DealSetConverter._four_hands is None:
            DealSetConverter._four_hands = DealSetConverter._compute_four_hands()
        return DealSet(self._lift(hs.bdd) & DealSetConverter._four_hands)

    def _lift(self, bdd: BDD) -> BDD:
        split = bdd.split()
        if split is True or split is False:
            return bdd
        if bdd in self._cache:
            return self._cache[bdd]

        var, avec, sans = split
        avec = self._lift(avec)
        sans = self._lift(sans)

        pi = self._pi
        n2 = BDD(2*var+1).thenelse(avec if pi & 2 else sans,
                                    sans if pi & 2 else avec)
        n1 = BDD(2*var).thenelse(n2 if pi & 1 else sans,
                                  sans if pi & 1 else n2)
        self._cache[bdd] = n1
        return n1

    @staticmethod
    def _compute_four_hands(N: int = 13) -> BDD:
        def sub(t, i):
            return tuple(x - (j == i) for j, x in enumerate(t))

        tier = {(0, 0, 0, 0): BDD.true()}
        for k in range(1, N*4 + 1):
            next_tier: dict = {}
            v1 = (N*4 - k) * 2
            v2 = v1 + 1
            for tupe in itertools.combinations(range(k + 3), 3):
                target = _tuple_to_pattern(tupe, k)
                if max(target) > N:
                    continue
                subs = [tier.get(sub(target, i), BDD.false()) for i in range(4)]
                tx = BDD(v2).thenelse(subs[2], subs[3])
                fx = BDD(v2).thenelse(subs[0], subs[1])
                next_tier[target] = BDD(v1).thenelse(tx, fx)
            tier = next_tier

        assert len(tier) == 1
        return tier[(N, N, N, N)]



# ---------------------------------------------------------------------------
# Shape parsing
# ---------------------------------------------------------------------------

class _IncrTuple(tuple):
    def __new__(cls, lst):
        return super().__new__(cls, tuple(lst))

    def incr(self, index):
        l = list(self)
        l[index] += 1
        return _IncrTuple(l)


class ShapeMaker:
    _BDDS: dict | None = None

    @staticmethod
    def _pattern_bdds() -> dict:
        if ShapeMaker._BDDS is not None:
            return ShapeMaker._BDDS

        SID = {"S": 0, "H": 1, "D": 2, "C": 3}
        so_far = {_IncrTuple([0, 0, 0, 0]): BDD.true()}
        for i, card in reversed(list(enumerate(_BDD_CARDS))):
            sid = SID[card.suit]
            var = BDD(i)
            after = {pat: (bdd & ~var) for pat, bdd in so_far.items()}
            for pat, bdd in so_far.items():
                ipat = pat.incr(sid)
                if sum(ipat) > 13:
                    continue
                ibdd = bdd & var
                after[ipat] = after[ipat] | ibdd if ipat in after else ibdd
            so_far = after

        ShapeMaker._BDDS = {pat: bdd for pat, bdd in so_far.items() if sum(pat) == 13}
        return ShapeMaker._BDDS

    @staticmethod
    def get_handset(spec: str) -> HandSet:
        m = hand_makers
        out = HandSet(BDD.false())
        for pat in parse_shape_spec(spec):
            pat_bdd = functools.reduce(
                HandSet.__and__,
                [(m.NUM_SP == pat[0]), (m.NUM_HE == pat[1]),
                 (m.NUM_DI == pat[2]), (m.NUM_CL == pat[3])]
            )
            out |= pat_bdd
        return out


# ---------------------------------------------------------------------------
# OrderedLengthMetric
# ---------------------------------------------------------------------------

class OrderedLengthMetric(HandSetMetric):
    def __init__(self, place: int):
        values: dict[int, BDD] = {}
        for pat, bdd in ShapeMaker._pattern_bdds().items():
            x = sorted(pat)[place]
            values[x] = values[x] | bdd if x in values else bdd
        super().__init__(values)


class SuitLengthCountMetric(HandSetMetric):
    """Counts the number of suits with exactly *target_length* cards."""
    def __init__(self, target_length: int):
        values: dict[int, BDD] = {}
        for pat, bdd in ShapeMaker._pattern_bdds().items():
            x = sum(1 for length in pat if length == target_length)
            values[x] = values[x] | bdd if x in values else bdd
        super().__init__(values)


# ---------------------------------------------------------------------------
# Lazy class-level constant
# ---------------------------------------------------------------------------

class lazy_const:
    def __init__(self, factory):
        self._factory = factory
        self._ready = False

    def __get__(self, obj, owner):
        if not self._ready:
            self._value = self._factory()
            self._ready = True
        return self._value


# ---------------------------------------------------------------------------
# hand_makers
# ---------------------------------------------------------------------------

def _count_spec_to_card_dict(spec: str) -> dict:
    mask = _parse_count_spec(spec)
    return {
        Card(suit, rank): 1
        for suit, offset in _SUIT_OFFSET.items()
        for rank, idx in _RANK_INDEX.items()
        if mask & (1 << (offset + idx))
    }


class hand_makers:
    """
    Factory for hand/deal constraints.  All attributes are lazily computed
    HandSetMetric or HandSet instances that can be combined with & | ~ and
    compared with == != < <= > >=.

    Example::

        m = hand_makers
        north_1nt = m.NORTH(
            (m.HCP >= 15) & (m.HCP <= 17) & m.SHAPE("any 4333 + any 5332 + any 4432")
        )
        df = north_1nt.sample_df(1000)
    """

    # Suit length metrics
    NUM_CL = lazy_const(lambda: SimpleHandMetric({Card("C", r): 1 for r in "AKQJT98765432"}))
    NUM_DI = lazy_const(lambda: SimpleHandMetric({Card("D", r): 1 for r in "AKQJT98765432"}))
    NUM_HE = lazy_const(lambda: SimpleHandMetric({Card("H", r): 1 for r in "AKQJT98765432"}))
    NUM_SP = lazy_const(lambda: SimpleHandMetric({Card("S", r): 1 for r in "AKQJT98765432"}))
    CLUBS    = NUM_CL
    DIAMONDS = NUM_DI
    HEARTS   = NUM_HE
    SPADES   = NUM_SP

    # Honor-based metrics
    HCP      = lazy_const(lambda: SimpleHandMetric(
        {Card(s, r): v for s in "SHDC" for r, v in [("A",4),("K",3),("Q",2),("J",1)]}))
    AKQ_POINTS = lazy_const(lambda: SimpleHandMetric(
        {Card(s, r): v for s in "SHDC" for r, v in [("A",3),("K",2),("Q",1)]}))
    CONTROLS = lazy_const(lambda: SimpleHandMetric(
        {Card(s, r): v for s in "SHDC" for r, v in [("A",2),("K",1)]}))
    ACES   = lazy_const(lambda: SimpleHandMetric({Card(s,"A"): 1 for s in "SHDC"}))
    KINGS  = lazy_const(lambda: SimpleHandMetric({Card(s,"K"): 1 for s in "SHDC"}))
    QUEENS = lazy_const(lambda: SimpleHandMetric({Card(s,"Q"): 1 for s in "SHDC"}))
    JACKS  = lazy_const(lambda: SimpleHandMetric({Card(s,"J"): 1 for s in "SHDC"}))
    TENS   = lazy_const(lambda: SimpleHandMetric({Card(s,"T"): 1 for s in "SHDC"}))
    TOP2   = lazy_const(lambda: SimpleHandMetric({Card(s,r): 1 for s in "SHDC" for r in "AK"}))
    TOP3   = lazy_const(lambda: SimpleHandMetric({Card(s,r): 1 for s in "SHDC" for r in "AKQ"}))
    TOP4   = lazy_const(lambda: SimpleHandMetric({Card(s,r): 1 for s in "SHDC" for r in "AKQJ"}))
    TOP5   = lazy_const(lambda: SimpleHandMetric({Card(s,r): 1 for s in "SHDC" for r in "AKQJT"}))

    QUICK_TRICKS_X2 = lazy_const(lambda: QuickTricksMetric())

    # Ordered shape
    LONGEST_SUIT        = lazy_const(lambda: OrderedLengthMetric(3))
    SECOND_SUIT         = lazy_const(lambda: OrderedLengthMetric(2))
    SHORTEST_SUIT       = lazy_const(lambda: OrderedLengthMetric(0))

    # Suit-count shape
    VOIDS      = lazy_const(lambda: SuitLengthCountMetric(0))
    SINGLETONS = lazy_const(lambda: SuitLengthCountMetric(1))
    DOUBLETONS = lazy_const(lambda: SuitLengthCountMetric(2))

    ALL_HANDS = lazy_const(lambda: HandSet(BDD.true()))

    @staticmethod
    def ANY(suit: str) -> HandSet:
        """HandSet where the hand holds at least one card in *suit* ('S', 'H', 'D', or 'C')."""
        return hand_makers.NUM(suit) >= 1

    # Direction converters
    WEST  = DealSetConverter("W")
    NORTH = DealSetConverter("N")
    EAST  = DealSetConverter("E")
    SOUTH = DealSetConverter("S")

    @staticmethod
    def NUM(spec: str) -> SimpleHandMetric:
        """Count of cards matching *spec* — same syntax as Hand.num().

        Suits (SHDC) then ranks (AKQJT2-9), comma-separated tokens.
        Examples: ``h.NUM("A")`` counts aces; ``h.NUM("HDK")`` counts red kings;
        ``h.NUM("A,SK")`` counts aces plus the king of spades.
        """
        return SimpleHandMetric(_count_spec_to_card_dict(spec))

    @staticmethod
    def HAS(card) -> HandSet:
        """HandSet for holding a specific card.  card can be a Card or 'SK' string."""
        if isinstance(card, str):
            card = Card(card[0], card[1])
        return HandSet(BDD(_BDD_VAR[card]))

    @staticmethod
    def MATCH_SHAPE(spec: str) -> HandSet:
        """
        Shape constraint string.  Examples:
          "4432"           – exactly 4S 4H 3D 2C
          "any 4432"       – any permutation of 4-4-3-2
          "44xx"           – 4S 4H any D any C
          "4432 + 4333"    – either shape
          "44xx - 4450"    – 4-4 majors, not 4-4-5-0
        """
        return ShapeMaker.get_handset(spec)

    @staticmethod
    def GOOD_SUIT(spec: str, suit: str) -> HandSet:
        """
        Return the HandSet where *suit* satisfies at least one pattern in the
        comma-separated *spec* — same semantics as ``Hand.good_suit(spec, suit)``.

        Example: ``h.GOOD_SUIT("AJx,KQx", "H")`` – hearts headed by AJ+ or KQ+.
        """
        rank_key = {k: i for i, k in enumerate("AKQJT98765432X")}
        suit_vars = [(v, c) for v, c in enumerate(_BDD_CARDS) if c.suit == suit]

        def one(s):
            s = s.strip().upper()
            spec_sorted = sorted(s, key=rank_key.__getitem__)
            states = [BDD.false()] * len(spec_sorted) + [BDD.true()]
            for var, card in reversed(suit_vars):
                new = list(states)
                for i, req in enumerate(spec_sorted):
                    if rank_key[req] >= rank_key[card.rank]:
                        new[i] = BDD(var).thenelse(new[i+1], new[i]) | new[i]
                states = new
            return HandSet(states[0])

        patterns = [p for p in spec.split(",") if p.strip()]
        return functools.reduce(HandSet.__or__, [one(p) for p in patterns])


h = hand_makers

__all__ = ["hand_makers", "h", "HandSet", "DealSet", "Card"]
