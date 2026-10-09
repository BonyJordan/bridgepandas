import numpy as np

from .auction import Contract, DeclaredContract
from .direction import Direction, TableVuln


def is_declarer_vulnerable(declarer, table_vuln) -> bool:
    """
    Return whether *declarer* is vulnerable.

    *declarer* is a Direction or W/N/E/S string.
    *table_vuln* is a TableVuln object or any string accepted by TableVuln() (-, e, n, b, ew, ns, both, …).
    """
    d = Direction(declarer)
    v = TableVuln(table_vuln)
    return v.ns_vul() if d.is_ns() else v.ew_vul()


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

def score(contract, tricks: int, is_vulnerable: bool) -> int:
    """
    Return the declarer's score for making *tricks* tricks in *contract*.

    *contract* is a Contract, DeclaredContract, Bid, or string like "3Nx".
    *tricks* is the total tricks taken (0–13).
    *is_vulnerable* is a bool.
    """
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


__all__ = [
    "is_declarer_vulnerable",
    "score",
    "scorediff_imps",
    "scorediff_matchpoints",
]
