import math
import pytest

from bridgepandas.handset import hand_makers, HandSet, DealSet
from bridgepandas.hand import int_to_hand_str, BridgeHandArray


m = hand_makers


# ---------------------------------------------------------------------------
# HandSetMetric comparisons and counts
# ---------------------------------------------------------------------------

class TestHandSetMetric:
    def test_hcp_exactly_0_count(self):
        # 0 HCP means no A/K/Q/J; choose 13 from the 36 non-honor cards
        import math
        assert (m.HCP == 0).count() == math.comb(36, 13)

    def test_hcp_exactly_37(self):
        # AKQJ in all four suits = 40; subtract 3 jacks = 37
        # Many possible hands; just confirm it's > 0
        assert (m.HCP == 37).count() > 0

    def test_hcp_ge_and_le_partition(self):
        # Count of hands with HCP in [15,17] = sum of individual equalities
        combined = (m.HCP >= 15) & (m.HCP <= 17)
        by_value = sum((m.HCP == n).count() for n in range(15, 18))
        assert combined.count() == by_value

    def test_suit_length_all_13(self):
        # Exactly one hand has all 13 spades
        assert (m.NUM_SP == 13).count() == 1

    def test_suit_length_zero(self):
        # Void in spades: must place 13 cards among the other 39
        assert (m.NUM_SP == 0).count() == math.comb(39, 13)

    def test_add_metrics(self):
        # NUM_SP + NUM_HE = total cards in majors
        majors = m.NUM_SP + m.NUM_HE
        # At least one hand with 13 cards in majors (all spades + all hearts impossible
        # in a 13-card hand, but 6+7 or 7+6 etc. are possible)
        assert (majors == 13).count() > 0

    def test_arithmetic_commutativity(self):
        # HCP + 0 should equal HCP (adding a zero metric)
        zero = m.HCP - m.HCP  # always 0, same BDD structure
        same = m.HCP + zero
        for n in range(0, 41):
            assert (same == n).count() == (m.HCP == n).count()


# ---------------------------------------------------------------------------
# HandSet boolean ops and sampling
# ---------------------------------------------------------------------------

class TestHandSet:
    def test_and_reduces_count(self):
        h15 = m.HCP >= 15
        sp5 = m.NUM_SP >= 5
        both = h15 & sp5
        assert both.count() < h15.count()
        assert both.count() < sp5.count()

    def test_or_increases_count(self):
        sp5 = m.NUM_SP >= 5
        he5 = m.NUM_HE >= 5
        either = sp5 | he5
        assert either.count() > sp5.count()

    def test_invert(self):
        sp5 = m.NUM_SP >= 5
        not_sp5 = ~sp5
        total = math.comb(52, 13)
        assert sp5.count() + not_sp5.count() == total

    def test_sample_satisfies_constraint(self):
        h15 = m.HCP >= 15
        for _ in range(10):
            hand = h15.sample()
            assert h15.contains(hand)

    def test_contains_known_hand(self):
        # AKQJ of spades + 9 random small cards: 10 HCP
        hand_str = "AKQJ/T98/765/432"
        from bridgepandas.hand import hand_str_to_int
        hand = hand_str_to_int(hand_str)
        assert (m.HCP == 10).contains(hand)
        assert not (m.HCP == 11).contains(hand)

    def test_sample_is_13_cards(self):
        hs = m.HCP >= 10
        for _ in range(5):
            hand = hs.sample()
            assert bin(hand).count("1") == 13


# ---------------------------------------------------------------------------
# Shape constraints
# ---------------------------------------------------------------------------

class TestShape:
    def test_4333_count(self):
        # C(52,13) total hands; 4333 is a specific shape
        s4333 = m.MATCH_SHAPE("4333")
        assert s4333.count() > 0
        # Any permutation should be 4x the specific shape
        assert m.MATCH_SHAPE("any 4333").count() == 4 * s4333.count()

    def test_4432_any(self):
        specific = m.MATCH_SHAPE("4432")
        any_4432 = m.MATCH_SHAPE("any 4432")
        # 4432 has 4!/(2!*1!*1!) = 12 distinct permutations
        assert any_4432.count() == 12 * specific.count()

    def test_addition(self):
        a = m.MATCH_SHAPE("4333")
        b = m.MATCH_SHAPE("any 4432")
        combined = m.MATCH_SHAPE("4333 + any 4432")
        assert combined.count() == a.count() + b.count()

    def test_subtraction(self):
        any_44xx = m.MATCH_SHAPE("any 44xx")
        no_4450  = m.MATCH_SHAPE("any 44xx - 4450 - 0445 - 5044 - 4504")
        assert no_4450.count() < any_44xx.count()

    def test_shape_sample_satisfies(self):
        shape = m.MATCH_SHAPE("any 5332")
        for _ in range(5):
            hand = shape.sample()
            lengths = sorted([
                bin(hand & ((1<<13)-1)).count("1"),
                bin((hand >> 13) & ((1<<13)-1)).count("1"),
                bin((hand >> 26) & ((1<<13)-1)).count("1"),
                bin((hand >> 39) & ((1<<13)-1)).count("1"),
            ])
            assert lengths == [2, 3, 3, 5]


# ---------------------------------------------------------------------------
# DealSet / DealSetConverter
# ---------------------------------------------------------------------------

class TestDealSet:
    def _north_1nt(self):
        bal = m.MATCH_SHAPE("any 4333 + any 5332 + any 4432")
        return m.NORTH((m.HCP >= 15) & (m.HCP <= 17) & bal)

    def test_count_is_large(self):
        ds = self._north_1nt()
        # Should be in the billions of billions
        assert ds.count() > 10**24

    def test_sample_df_shape(self):
        ds = self._north_1nt()
        df = ds.sample_df(10, seed=1)
        assert df.shape == (10, 4)
        assert list(df.columns) == ["west", "north", "east", "south"]

    def test_sample_df_dtypes(self):
        ds = self._north_1nt()
        df = ds.sample_df(5, seed=1)
        for col in df.columns:
            assert str(df[col].dtype) == "BridgeHand"

    def test_sample_df_north_satisfies_constraint(self):
        bal = m.MATCH_SHAPE("any 4333 + any 5332 + any 4432")
        nt_constraint = (m.HCP >= 15) & (m.HCP <= 17) & bal
        ds = m.NORTH(nt_constraint)
        df = ds.sample_df(10, seed=2)
        for north_int in df["north"].array._data:
            assert nt_constraint.contains(int(north_int))

    def test_sample_df_all_52_cards(self):
        ds = self._north_1nt()
        df = ds.sample_df(10, seed=3)
        for _, row in df.iterrows():
            combined = row["west"] | row["north"] | row["east"] | row["south"]
            assert combined == (1 << 52) - 1

    def test_contains_round_trip(self):
        ds = self._north_1nt()
        deal = ds.sample()
        assert ds.contains(deal["west"], deal["north"], deal["east"], deal["south"])

    def test_set_operations(self):
        bal = m.MATCH_SHAPE("any 4333 + any 5332 + any 4432")
        n15 = m.NORTH((m.HCP >= 15) & (m.HCP <= 17) & bal)
        n12 = m.NORTH((m.HCP >= 12) & (m.HCP <= 14) & bal)
        union = n15 | n12
        assert union.count() == n15.count() + n12.count()
