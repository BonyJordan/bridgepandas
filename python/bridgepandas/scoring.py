import numpy as np
import pandas as pd

from .auction import Contract, DeclaredContract
from .direction import Direction, TableVuln, _factorize_attrs


def is_declarer_vulnerable(declarer, table_vuln) -> bool:
    """
    Return whether *declarer* is vulnerable.

    *declarer* is a Direction or W/N/E/S string.
    *table_vuln* is a TableVuln object or any string accepted by TableVuln() (-, e, n, b, ew, ns, both, …).

    Either argument may instead be a pandas Series, in which case the result
    is a pandas Series of bool with the same length (the Index is taken from
    *declarer* if it is a Series, else from *table_vuln*).
    """
    declarer_is_series = isinstance(declarer, pd.Series)
    vuln_is_series = isinstance(table_vuln, pd.Series)

    if not declarer_is_series and not vuln_is_series:
        d = Direction(declarer)
        v = TableVuln(table_vuln)
        return v.ns_vul() if d.is_ns() else v.ew_vul()

    if declarer_is_series:
        is_ns = _factorize_attrs(declarer, Direction, is_ns=Direction.is_ns)["is_ns"]
    else:
        is_ns = Direction(declarer).is_ns()

    if vuln_is_series:
        attrs = _factorize_attrs(table_vuln, TableVuln, ns_vul=TableVuln.ns_vul, ew_vul=TableVuln.ew_vul)
        ns_vul, ew_vul = attrs["ns_vul"], attrs["ew_vul"]
    else:
        v = TableVuln(table_vuln)
        ns_vul, ew_vul = v.ns_vul(), v.ew_vul()

    index = declarer.index if declarer_is_series else table_vuln.index
    return pd.Series(np.where(is_ns, ns_vul, ew_vul), index=index)


_IMPS_TABLE = np.array([
    15, 45, 85, 125, 165, 215, 265, 315, 365,
    425, 495, 595, 745, 895, 1095, 1295, 1495, 1745, 1995,
    2245, 2495, 2995, 3495, 3995,
])


def scorediff_imps(diff):
    """Convert (my_score - their_score) to IMPs.

    *diff* may be a scalar int or a pandas Series/array of ints, in which
    case the result has the same shape (e.g. ``(df['score_a'] -
    df['score_b']).map(...)`` works, but so does plain subtraction:
    ``scorediff_imps(df['score_a'] - df['score_b'])``).
    """
    magnitude = np.searchsorted(_IMPS_TABLE, np.abs(diff), side="left")
    result = np.sign(diff) * magnitude
    return int(result) if np.ndim(diff) == 0 else result


def scorediff_matchpoints(diff):
    """Convert (my_score - their_score) to matchpoints on a 0/0.5/1 scale.

    *diff* may be a scalar int or a pandas Series/array of ints, in which
    case the result has the same shape.
    """
    result = (np.sign(diff) + 1) / 2
    return float(result) if np.ndim(diff) == 0 else result


def score_ns(declared_contract: str|DeclaredContract, declarer_tricks: int,
             table_vulnerable: str|TableVuln) -> int:
    dc = DeclaredContract(declared_contract)
    vul = TableVuln(table_vulnerable)
    dec_score = score(dc, declarer_tricks, vul.is_vul(dc.declarer))
    if dc.declarer.is_ew():
        return -dec_score
    else:
        return dec_score

def _score_array(level, strain, double_state, tricks_needed, tricks, is_vulnerable):
    """Array-aware re-implementation of score()'s branching, operating on
    already-extracted Contract fields (each either a python scalar or a
    numpy array of matching length) plus tricks/is_vulnerable (scalar or
    array-like). Returns a numpy int64 array."""
    tricks = np.asarray(tricks)
    is_vulnerable = np.asarray(is_vulnerable, dtype=bool)
    level = np.asarray(level)
    double_state = np.asarray(double_state)
    tricks_needed = np.asarray(tricks_needed)
    strain = np.asarray(strain)

    shortfall = tricks_needed - tricks
    overtricks = tricks - tricks_needed

    # ---- down ----
    down_undoubled = np.where(is_vulnerable, -100 * shortfall, -50 * shortfall)
    down_doubled_vul = double_state * (100 - 300 * shortfall)
    down_doubled_nv = np.where(shortfall < 4,
                                double_state * (100 - 200 * shortfall),
                                double_state * (400 - 300 * shortfall))
    down_doubled = np.where(is_vulnerable, down_doubled_vul, down_doubled_nv)
    down_score = np.where(double_state == 0, down_undoubled, down_doubled)

    # ---- made ----
    is_notrump = strain == "N"
    is_major = np.isin(strain, ("S", "H"))
    btl = np.where(is_notrump, 10 + 30 * level,
                   np.where(is_major, 30 * level, 20 * level))
    btl = btl * (2 ** double_state)

    bonus = np.where(level == 7, np.where(is_vulnerable, 2000, 1300),
             np.where(level == 6, np.where(is_vulnerable, 1250, 800),
             np.where(btl >= 100, np.where(is_vulnerable, 500, 300),
                      np.full_like(btl, 50))))
    bonus = bonus + 50 * double_state  # insult bonus

    is_minor = np.isin(strain, ("C", "D"))
    overtrick_doubled = overtricks * double_state * np.where(is_vulnerable, 200, 100)
    overtrick_undoubled = np.where(is_minor, overtricks * 20, overtricks * 30)
    bonus = bonus + np.where(double_state > 0, overtrick_doubled, overtrick_undoubled)

    made_score = btl + bonus
    return np.where(tricks < tricks_needed, down_score, made_score).astype(np.int64)


def score(contract, tricks: int, is_vulnerable: bool) -> int:
    """
    Return the declarer's score for making *tricks* tricks in *contract*.

    *contract* is a Contract, DeclaredContract, Bid, or string like "3Nx".
    *tricks* is the total tricks taken (0–13).
    *is_vulnerable* is a bool.

    Any of *contract*, *tricks*, or *is_vulnerable* may instead be a pandas
    Series (or, for *tricks*/*is_vulnerable*, any array-like) — the result
    then has the same shape (e.g. ``score(df['contract'], df['tricks'],
    df['is_vul'])`` works, and so does mixing a scalar *contract* with a
    Series *tricks*). The Index of the result is taken from the first of
    *contract*, *tricks*, *is_vulnerable* (in that order) that is a pandas
    Series.
    """
    contract_is_series = isinstance(contract, pd.Series)
    vectorized = contract_is_series or np.ndim(tricks) > 0 or np.ndim(is_vulnerable) > 0

    if not vectorized:
        con = Contract(contract)

        if tricks < con.tricks_needed:
            shortfall = con.tricks_needed - tricks
            if is_vulnerable:
                if con.double_state == 0:
                    return -100 * shortfall
                else:
                    return con.double_state * (100 - 300 * shortfall)
            else:
                if con.double_state == 0:
                    return -50 * shortfall
                elif shortfall < 4:
                    return con.double_state * (100 - 200 * shortfall)
                else:
                    return con.double_state * (400 - 300 * shortfall)

        # Made the contract
        if con.strain in "Nn":
            btl = 10 + 30 * con.level
        elif con.strain in "SsHh":
            btl = 30 * con.level
        else:
            btl = 20 * con.level

        btl *= 2 ** con.double_state

        if con.level == 7:
            bonus = 2000 if is_vulnerable else 1300
        elif con.level == 6:
            bonus = 1250 if is_vulnerable else 800
        elif btl >= 100:
            bonus = 500 if is_vulnerable else 300
        else:
            bonus = 50

        bonus += 50 * con.double_state  # insult bonus

        overtricks = tricks - con.tricks_needed
        if con.double_state > 0:
            bonus += overtricks * con.double_state * (200 if is_vulnerable else 100)
        elif con.strain in "CcDd":
            bonus += overtricks * 20
        else:
            bonus += overtricks * 30

        return btl + bonus

    if contract_is_series:
        attrs = _factorize_attrs(
            contract, Contract,
            level=lambda c: c.level, strain=lambda c: c.strain,
            double_state=lambda c: c.double_state, tricks_needed=lambda c: c.tricks_needed,
        )
        level, strain, double_state, tricks_needed = (
            attrs["level"], attrs["strain"], attrs["double_state"], attrs["tricks_needed"])
    else:
        con = Contract(contract)
        level, strain, double_state, tricks_needed = (
            con.level, con.strain, con.double_state, con.tricks_needed)

    result = _score_array(level, strain, double_state, tricks_needed, tricks, is_vulnerable)

    if contract_is_series:
        index = contract.index
    elif isinstance(tricks, pd.Series):
        index = tricks.index
    elif isinstance(is_vulnerable, pd.Series):
        index = is_vulnerable.index
    else:
        index = None

    return pd.Series(result, index=index)


__all__ = [
    "is_declarer_vulnerable",
    "score",
    "scorediff_imps",
    "scorediff_matchpoints",
]
