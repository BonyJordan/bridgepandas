import numpy as np
import pandas as pd
import pytest

from bridgepandas.hand import (
    hand_str_to_int,
    int_to_hand_str,
    BridgeHandArray,
    BridgeHandDtype,
    random_deals,
)


# ---------------------------------------------------------------------------
# Encoding round-trips
# ---------------------------------------------------------------------------

class TestEncoding:
    @pytest.mark.parametrize("hand", [
        "AKQJT98765432/-/-/-",  # all spades
        "-/AKQJT98765432/-/-",  # all hearts
        "-/-/AKQJT98765432/-",  # all diamonds
        "-/-/-/AKQJT98765432",  # all clubs
        "QJ6/K652/J85/T98",     # example from spec
        "AKQ/JT9/876/543",
        "-/-/-/AKQJT98765432",
    ])
    def test_round_trip(self, hand):
        assert int_to_hand_str(hand_str_to_int(hand)) == hand

    def test_void_suit_display(self):
        hand = "AK/-/AK/-"
        assert int_to_hand_str(hand_str_to_int(hand)) == hand

    def test_cards_are_high_to_low(self):
        # Encoding should sort within suit high-to-low
        encoded = hand_str_to_int("62AKQ/432/432/432")  # unsorted spades
        assert int_to_hand_str(encoded) == "AKQ62/432/432/432"

    def test_duplicate_card_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            hand_str_to_int("AA/-/KQJ98765432/-")

    def test_wrong_suit_count_raises(self):
        with pytest.raises(ValueError):
            hand_str_to_int("AKQ/JT9/876")

    def test_bad_rank_raises(self):
        with pytest.raises(ValueError, match="Unknown rank"):
            hand_str_to_int("1/-/-/-")

    def test_13_cards_total(self):
        hand = "QJ6/K652/J85/T98"
        encoded = hand_str_to_int(hand)
        assert bin(encoded).count("1") == 13


# ---------------------------------------------------------------------------
# BridgeHandArray
# ---------------------------------------------------------------------------

class TestBridgeHandArray:
    def test_dtype_name(self):
        arr = pd.array(["AKQ/JT9/876/543"], dtype="BridgeHand")
        assert arr.dtype.name == "BridgeHand"
        assert isinstance(arr.dtype, BridgeHandDtype)

    def test_from_strings(self):
        arr = pd.array(["AKQ/JT9/876/543", "QJ6/K652/J85/T98"], dtype="BridgeHand")
        assert len(arr) == 2

    def test_na_handling(self):
        arr = pd.array(["AKQ/JT9/876/543", None, "QJ6/K652/J85/T98"], dtype="BridgeHand")
        assert arr.isna().tolist() == [False, True, False]
        assert arr[1] is pd.NA

    def test_getitem_scalar(self):
        arr = pd.array(["QJ6/K652/J85/T98"], dtype="BridgeHand")
        val = arr[0]
        assert isinstance(val, int)
        assert int_to_hand_str(val) == "QJ6/K652/J85/T98"

    def test_getitem_slice(self):
        arr = pd.array(["AKQ/JT9/876/543", "QJ6/K652/J85/T98", None], dtype="BridgeHand")
        sliced = arr[1:]
        assert isinstance(sliced, BridgeHandArray)
        assert len(sliced) == 2

    def test_concat(self):
        a = pd.array(["AKQ/JT9/876/543"], dtype="BridgeHand")
        b = pd.array(["QJ6/K652/J85/T98"], dtype="BridgeHand")
        result = BridgeHandArray._concat_same_type([a, b])
        assert len(result) == 2

    def test_dataframe_column(self):
        df = pd.DataFrame({"hand": pd.array(["AKQ/JT9/876/543", None], dtype="BridgeHand")})
        assert str(df.dtypes["hand"]) == "BridgeHand"
        assert df["hand"].isna().sum() == 1

    def test_to_strings(self):
        arr = pd.array(["AKQ/JT9/876/543", None], dtype="BridgeHand")
        result = arr.to_strings()
        assert result[0] == "AKQ/JT9/876/543"
        assert result[1] is pd.NA


# ---------------------------------------------------------------------------
# .hcp accessor
# ---------------------------------------------------------------------------

class TestHcpAccessor:
    def make_series(self, *hands):
        return pd.Series(pd.array(list(hands), dtype="BridgeHand"))

    def test_akqj_is_10(self):
        s = self.make_series("AKQJ/-/-/-")
        assert s.hcp.tolist() == [10]

    def test_no_honors_is_0(self):
        s = self.make_series("T98765432/-/-/-")
        assert s.hcp.tolist() == [0]

    def test_full_deck_honors(self):
        # 4 aces + 4 kings + 4 queens + 4 jacks = 16+12+8+4 = 40
        s = self.make_series("AKQJ/AKQJ/AKQJ/AKQJ")
        assert s.hcp.iloc[0] == 40

    def test_na_propagates(self):
        s = self.make_series("AKQJ/-/-/-", None)
        assert s.hcp.isna().tolist() == [False, True]

    def test_comparison_returns_bool_series(self):
        s = self.make_series("AKQJ/-/-/-", "T98765432/-/-/-")
        result = s.hcp >= 10
        assert result.tolist() == [True, False]

    def test_arithmetic(self):
        s = self.make_series("AK/-/-/-", "QJ/-/-/-")
        total = s.hcp + s.hcp  # same series, just checking arithmetic works
        assert total.tolist() == [14, 6]


# ---------------------------------------------------------------------------
# Suit-length accessors
# ---------------------------------------------------------------------------

class TestSuitAccessors:
    def make_series(self, hand):
        return pd.Series(pd.array([hand], dtype="BridgeHand"))

    @pytest.mark.parametrize("hand,expected", [
        ("AKQJT98765432/-/-/-", (13, 0, 0, 0)),
        ("-/AKQJT98765432/-/-", (0, 13, 0, 0)),
        ("-/-/AKQJT98765432/-", (0, 0, 13, 0)),
        ("-/-/-/AKQJT98765432", (0, 0, 0, 13)),
        ("AKQ/JT9/876/543",     (3, 3, 3, 3)),
        ("AKQJ/T98/765/432",    (4, 3, 3, 3)),
    ])
    def test_lengths(self, hand, expected):
        s = self.make_series(hand)
        assert (s.spades.iloc[0], s.hearts.iloc[0],
                s.diamonds.iloc[0], s.clubs.iloc[0]) == expected

    def test_sum_to_13(self):
        hands = ["QJ6/K652/J85/T98", "AKQ/JT98/876/543", "AKQJT/-/98765432/-"]
        s = pd.Series(pd.array(hands, dtype="BridgeHand"))
        total = s.spades + s.hearts + s.diamonds + s.clubs
        assert (total == 13).all()

    def test_pattern_and_handshape(self):
        hands = ["QJ6/K652/J85/T98", "AKQJT/-/98765432/-", None]
        s = pd.Series(pd.array(hands, dtype="BridgeHand"))
        assert s.pattern.tolist() == [(3, 4, 3, 3), (5, 0, 8, 0), None]
        assert s.handshape.tolist() == [(4, 3, 3, 3), (8, 5, 0, 0), None]

    def test_pattern_matches_scalar(self):
        from bridgepandas.hand import Hand
        df = random_deals(200, seed=0)
        expected = [Hand(int(v)).pattern for v in df.south.array._data]
        assert df.south.pattern.tolist() == expected
        assert df.south.pattern.value_counts().sum() == 200


# ---------------------------------------------------------------------------
# random_deals
# ---------------------------------------------------------------------------

class TestRandomDeals:
    def test_columns(self):
        df = random_deals(1)
        assert list(df.columns) == ["west", "north", "east", "south"]

    def test_dtypes(self):
        df = random_deals(1)
        for col in df.columns:
            assert str(df[col].dtype) == "BridgeHand"

    def test_all_52_cards_dealt(self):
        df = random_deals(20, seed=0)
        for _, row in df.iterrows():
            combined = row["west"] | row["north"] | row["east"] | row["south"]
            assert combined == (1 << 52) - 1

    def test_13_cards_per_hand(self):
        df = random_deals(20, seed=0)
        for _, row in df.iterrows():
            for col in ["west", "north", "east", "south"]:
                assert bin(row[col]).count("1") == 13

    def test_no_card_overlap(self):
        df = random_deals(20, seed=0)
        for _, row in df.iterrows():
            hands = [row[c] for c in ["west", "north", "east", "south"]]
            for i in range(4):
                for j in range(i + 1, 4):
                    assert hands[i] & hands[j] == 0

    def test_reproducible_with_seed(self):
        df1 = random_deals(5, seed=99)
        df2 = random_deals(5, seed=99)
        for col in df1.columns:
            assert df1[col].array._data.tolist() == df2[col].array._data.tolist()
