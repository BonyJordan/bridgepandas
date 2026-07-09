import pytest
import numpy as np

from bridgepandas.deal import Deal, random_deals
from bridgepandas.hand import hand_str_to_int, int_to_hand_str


# ---------------------------------------------------------------------------
# Deal construction and display
# ---------------------------------------------------------------------------

class TestDeal:
    def test_from_strings_round_trip(self):
        deal = Deal.from_strings(
            west="AKQ/JT9/876/543",
            north="JT9/876/543/AKQ",
            east="876/543/AKQ/JT9",
            south="543/AKQ/JT9/876",
        )
        assert int_to_hand_str(deal.west) == "AKQ/JT9/876/543"
        assert int_to_hand_str(deal.north) == "JT9/876/543/AKQ"

    def test_from_row(self):
        import pandas as pd
        from bridgepandas import BridgeHandArray
        df = random_deals(1, seed=0)
        deal = Deal.from_row(df.iloc[0])
        assert deal.west | deal.north | deal.east | deal.south == (1 << 52) - 1

    def test_to_dict(self):
        deal = Deal.from_strings("AKQ/JT9/876/543", "543/876/JT9/AKQ",
                                  "JT9/AKQ/543/876", "876/543/AKQ/JT9")
        d = deal.to_dict()
        assert set(d.keys()) == {"west", "north", "east", "south"}
        assert d["west"] == deal.west

    def test_to_dataframe(self):
        deals = [
            Deal.from_strings("AKQ/JT9/876/543", "543/876/JT9/AKQ",
                               "JT9/AKQ/543/876", "876/543/AKQ/JT9"),
            Deal.from_strings("JT9/876/543/AKQ", "AKQ/543/876/JT9",
                               "543/JT9/AKQ/876", "876/AKQ/JT9/543"),
        ]
        df = Deal.to_dataframe(deals)
        assert df.shape == (2, 4)
        assert list(df.columns) == ["west", "north", "east", "south"]
        assert str(df["west"].dtype) == "BridgeHand"

    def test_hashable(self):
        deal = Deal.from_strings("AKQ/JT9/876/543", "543/876/JT9/AKQ",
                                  "JT9/AKQ/543/876", "876/543/AKQ/JT9")
        s = {deal, deal}
        assert len(s) == 1

    def test_str_compass_layout(self):
        deal = Deal.from_strings("AKQ/JT9/876/543", "JT9/876/543/AKQ",
                                  "876/543/AKQ/JT9", "543/AKQ/JT9/876")
        s = str(deal)
        lines = s.splitlines()
        assert len(lines) == 3
        assert "N" in lines[0]
        assert "W" in lines[1] and "E" in lines[1]
        assert "S" in lines[2]


# ---------------------------------------------------------------------------
# random_deals — unconstrained
# ---------------------------------------------------------------------------

class TestRandomDealsUnconstrained:
    def test_shape(self):
        df = random_deals(10, seed=0)
        assert df.shape == (10, 4)
        assert list(df.columns) == ["west", "north", "east", "south"]

    def test_all_52_cards(self):
        df = random_deals(10, seed=1)
        for _, row in df.iterrows():
            assert row["west"] | row["north"] | row["east"] | row["south"] == (1 << 52) - 1

    def test_no_overlap(self):
        df = random_deals(10, seed=2)
        for _, row in df.iterrows():
            hands = [row[c] for c in ["west", "north", "east", "south"]]
            for i in range(4):
                for j in range(i + 1, 4):
                    assert hands[i] & hands[j] == 0

    def test_reproducible(self):
        df1 = random_deals(5, seed=77)
        df2 = random_deals(5, seed=77)
        for col in df1.columns:
            assert df1[col].array._data.tolist() == df2[col].array._data.tolist()


# ---------------------------------------------------------------------------
# random_deals — BDD fast path (HandSet specs)
# ---------------------------------------------------------------------------

class TestRandomDealsFast:
    def test_str_spec_cards_present(self):
        # West must have AS KS QS JS (AKQJ of spades)
        df = random_deals(10, west="AKQJ/-/-/-", seed=0)
        # AKQJ of spades = bits 39+12, 39+11, 39+10, 39+9
        mask = sum(1 << (39 + r) for r in [9, 10, 11, 12])
        for _, row in df.iterrows():
            assert (row["west"] & mask) == mask

    def test_str_spec_all_52_cards(self):
        df = random_deals(5, west="AKQJ/-/-/-", seed=1)
        for _, row in df.iterrows():
            assert row["west"] | row["north"] | row["east"] | row["south"] == (1 << 52) - 1

    def test_handset_spec(self):
        from bridgepandas.handset import hand_makers
        m = hand_makers
        nt = (m.HCP >= 15) & (m.HCP <= 17) & m.MATCH_SHAPE("any 4333 + any 5332 + any 4432")
        df = random_deals(10, north=nt, seed=3)
        for north_int in df["north"].array._data:
            assert nt.contains(int(north_int))

    def test_two_str_specs_no_overlap(self):
        # Give west and east each specific cards — no card overlap allowed
        df = random_deals(5, west="AKQJ/-/-/-", east="-/AKQJ/-/-", seed=4)
        for _, row in df.iterrows():
            assert row["west"] & row["east"] == 0

    def test_dtypes_are_bridgehand(self):
        df = random_deals(5, west="AK/-/-/-", seed=0)
        for col in df.columns:
            assert str(df[col].dtype) == "BridgeHand"


# ---------------------------------------------------------------------------
# random_deals — slow path (callables / accept)
# ---------------------------------------------------------------------------

class TestRandomDealsSlow:
    def test_callable_spec(self):
        # West must have at least 5 spades
        df = random_deals(10, west=lambda h: bin(h >> 39).count("1") >= 5, seed=5)
        assert df.shape == (10, 4)
        for _, row in df.iterrows():
            assert bin(row["west"] >> 39).count("1") >= 5

    def test_accept_function(self):
        # Accept only deals where north has more HCP than south
        def north_beats_south(deal):
            def hcp(h):
                pts = 0
                for suit in range(4):
                    suit_bits = (h >> (suit * 13)) & 0x1FFF
                    for rank, val in [(12, 4), (11, 3), (10, 2), (9, 1)]:
                        if suit_bits & (1 << rank):
                            pts += val
                return pts
            return hcp(deal.north) > hcp(deal.south)

        df = random_deals(5, accept=north_beats_south, seed=6)
        assert df.shape == (5, 4)
        for _, row in df.iterrows():
            deal = Deal.from_row(row)
            assert north_beats_south(deal)

    def test_all_52_cards_slow_path(self):
        df = random_deals(5, west=lambda h: True, seed=7)
        for _, row in df.iterrows():
            assert row["west"] | row["north"] | row["east"] | row["south"] == (1 << 52) - 1

    def test_fail_count_raises(self):
        # Impossible constraint: west has both AS and west doesn't have AS
        with pytest.raises(ValueError, match="No deals found"):
            random_deals(
                1,
                west=lambda h: False,  # nothing passes
                fail_count=10,
            )

    def test_mixed_spec_and_callable(self):
        # East gets QJ of spades via str spec, north must have >= 10 HCP via callable
        df = random_deals(5, east="QJ/-/-/-",
                          north=lambda h: bin(h).count("1") == 13,  # always true, forces slow path
                          seed=8)
        east_mask = (1 << (39 + 10)) | (1 << (39 + 9))  # Q=rank10, J=rank9 of spades
        for _, row in df.iterrows():
            assert (row["east"] & east_mask) == east_mask
