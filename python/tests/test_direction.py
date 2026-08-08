import pandas as pd
import pytest

from bridgepandas.direction import (
    Direction, TableVuln, board_number_to_dealer_vuln, dealer_vuln_to_board_number,
)


class TestBoardNumberToDealerVuln:
    @pytest.mark.parametrize("board,dealer,vuln", [
        (1, "N", "-"), (2, "E", "n"), (3, "S", "e"), (4, "W", "b"),
        (5, "N", "n"), (8, "W", "-"), (13, "N", "b"), (16, "W", "e"),
        (17, "N", "-"),  # wraps past 16 like board 1
    ])
    def test_known_boards(self, board, dealer, vuln):
        d, v = board_number_to_dealer_vuln(board)
        assert d == Direction(dealer)
        assert v == TableVuln(vuln)

    def test_series_matches_scalar_loop(self):
        boards = pd.Series(range(1, 17))
        dealers, vulns = board_number_to_dealer_vuln(boards)
        assert isinstance(dealers, pd.Series)
        assert isinstance(vulns, pd.Series)
        expected_d = [board_number_to_dealer_vuln(b)[0] for b in boards]
        expected_v = [board_number_to_dealer_vuln(b)[1] for b in boards]
        assert dealers.tolist() == expected_d
        assert vulns.tolist() == expected_v

    def test_index_preserved(self):
        boards = pd.Series([1, 5, 13], index=["a", "b", "c"])
        dealers, vulns = board_number_to_dealer_vuln(boards)
        assert dealers.index.tolist() == ["a", "b", "c"]
        assert vulns.index.tolist() == ["a", "b", "c"]


class TestDealerVulnToBoardNumber:
    @pytest.mark.parametrize("dealer,vuln,board", [
        ("N", "-", 1), ("E", "n", 2), ("S", "e", 3), ("W", "b", 4),
        ("N", "n", 5), ("W", "-", 8), ("N", "b", 13), ("W", "e", 16),
    ])
    def test_known(self, dealer, vuln, board):
        assert dealer_vuln_to_board_number(dealer, vuln) == board

    def test_round_trip_all_16(self):
        for b in range(1, 17):
            d, v = board_number_to_dealer_vuln(b)
            assert dealer_vuln_to_board_number(d, v) == b

    def test_both_series(self):
        dealers = pd.Series(["N", "E", "S", "W"])
        vulns = pd.Series(["-", "n", "e", "b"])
        expected = [dealer_vuln_to_board_number(d, v) for d, v in zip(dealers, vulns)]
        result = dealer_vuln_to_board_number(dealers, vulns)
        assert isinstance(result, pd.Series)
        assert result.tolist() == expected

    def test_series_dealer_scalar_vuln(self):
        dealers = pd.Series(["N", "E", "S", "W"])
        expected = [dealer_vuln_to_board_number(d, "b") for d in dealers]
        assert dealer_vuln_to_board_number(dealers, "b").tolist() == expected

    def test_index_prefers_dealer(self):
        dealers = pd.Series(["N", "E"], index=[1, 2])
        vulns = pd.Series(["-", "n"], index=[9, 8])
        assert dealer_vuln_to_board_number(dealers, vulns).index.tolist() == [1, 2]
