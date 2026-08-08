import pandas as pd
import pytest

from bridgepandas.auction import DeclaredContract
from bridgepandas.scoring import score, is_declarer_vulnerable, scorediff_imps, scorediff_matchpoints


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------

class TestScoreScalar:
    @pytest.mark.parametrize("contract,tricks,vul,expected", [
        ("3N", 9, False, 400),
        ("3N", 10, False, 430),
        ("4S", 10, True, 620),
        ("3N", 8, False, -50),
        ("4S", 9, False, -50),
        ("4S", 9, True, -100),
        ("4Sx", 10, False, 590),
        ("4Sx", 8, True, -500),
        ("7N", 13, True, 2220),
        ("1Cxx", 7, False, 230),
        ("2Hx", 6, True, -500),
        ("1N", 7, False, 90),
        ("6C", 12, True, 1370),
    ])
    def test_known_scores(self, contract, tricks, vul, expected):
        assert score(contract, tricks, vul) == expected

    def test_scalar_returns_python_int(self):
        assert type(score("3N", 9, False)) is int


class TestScoreSeries:
    CASES = [
        ("3N", 9, False), ("3N", 10, False), ("4S", 10, True),
        ("3N", 8, False), ("4Sx", 10, False), ("7N", 13, True),
        ("1Cxx", 7, False), ("2Hx", 6, True), ("4Sx", 8, True),
    ]

    def test_all_three_as_series_matches_scalar_loop(self):
        contracts, tricks, vuls = zip(*self.CASES)
        expected = [score(c, t, v) for c, t, v in self.CASES]
        result = score(pd.Series(contracts), pd.Series(tricks), pd.Series(vuls))
        assert isinstance(result, pd.Series)
        assert result.tolist() == expected

    def test_scalar_contract_series_tricks(self):
        tricks = pd.Series([9, 10, 8])
        expected = [score("3N", t, False) for t in tricks]
        result = score("3N", tricks, False)
        assert isinstance(result, pd.Series)
        assert result.tolist() == expected

    def test_series_contract_scalar_tricks_and_vul(self):
        contracts = pd.Series(["3N", "4S", "4Sx"])
        expected = [score(c, 9, True) for c in contracts]
        assert score(contracts, 9, True).tolist() == expected

    def test_declared_contract_objects_in_series(self):
        contracts = pd.Series([DeclaredContract("3N-S"), DeclaredContract("4Sx-N")])
        expected = [score(c, 9, False) for c in contracts]
        assert score(contracts, 9, False).tolist() == expected

    def test_index_preserved_from_contract(self):
        contracts = pd.Series(["3N", "4S"], index=[10, 20])
        result = score(contracts, pd.Series([9, 10]), False)
        assert result.index.tolist() == [10, 20]

    def test_index_falls_back_to_tricks_when_contract_scalar(self):
        tricks = pd.Series([9, 10], index=[5, 6])
        result = score("3N", tricks, False)
        assert result.index.tolist() == [5, 6]

    def test_dtype_is_integer(self):
        result = score(pd.Series(["3N", "4S"]), pd.Series([9, 10]), pd.Series([False, True]))
        assert pd.api.types.is_integer_dtype(result)


# ---------------------------------------------------------------------------
# is_declarer_vulnerable()
# ---------------------------------------------------------------------------

class TestIsDeclarerVulnerable:
    @pytest.mark.parametrize("declarer,vuln,expected", [
        ("N", "n", True), ("N", "e", False), ("E", "b", True), ("W", "-", False),
        ("S", "n", True), ("E", "e", True), ("W", "n", False),
    ])
    def test_scalar(self, declarer, vuln, expected):
        assert is_declarer_vulnerable(declarer, vuln) is expected

    def test_both_series(self):
        declarers = pd.Series(["N", "E", "S", "W"])
        vulns = pd.Series(["n", "b", "-", "e"])
        expected = [is_declarer_vulnerable(d, v) for d, v in zip(declarers, vulns)]
        result = is_declarer_vulnerable(declarers, vulns)
        assert isinstance(result, pd.Series)
        assert result.tolist() == expected
        assert result.dtype == bool

    def test_series_declarer_scalar_vuln(self):
        declarers = pd.Series(["N", "E", "S", "W"])
        expected = [is_declarer_vulnerable(d, "b") for d in declarers]
        assert is_declarer_vulnerable(declarers, "b").tolist() == expected

    def test_scalar_declarer_series_vuln(self):
        vulns = pd.Series(["n", "e", "b", "-"])
        expected = [is_declarer_vulnerable("N", v) for v in vulns]
        assert is_declarer_vulnerable("N", vulns).tolist() == expected

    def test_index_prefers_declarer(self):
        declarers = pd.Series(["N", "E"], index=[1, 2])
        vulns = pd.Series(["n", "e"], index=[9, 8])
        assert is_declarer_vulnerable(declarers, vulns).index.tolist() == [1, 2]


# ---------------------------------------------------------------------------
# scorediff_imps() / scorediff_matchpoints()
# ---------------------------------------------------------------------------

class TestScorediffImps:
    @pytest.mark.parametrize("diff,expected", [
        (400, 9), (-400, -9), (0, 0), (3000, 22), (-3000, -22),
    ])
    def test_scalar(self, diff, expected):
        result = scorediff_imps(diff)
        assert result == expected
        assert isinstance(result, int)

    def test_series(self):
        result = scorediff_imps(pd.Series([400, -400, 0, 3000]))
        assert isinstance(result, pd.Series)
        assert result.tolist() == [9, -9, 0, 22]


class TestScorediffMatchpoints:
    @pytest.mark.parametrize("diff,expected", [(400, 1.0), (-1, 0.0), (0, 0.5)])
    def test_scalar(self, diff, expected):
        result = scorediff_matchpoints(diff)
        assert result == expected
        assert isinstance(result, float)

    def test_series(self):
        result = scorediff_matchpoints(pd.Series([400, -1, 0]))
        assert isinstance(result, pd.Series)
        assert result.tolist() == [1.0, 0.0, 0.5]
