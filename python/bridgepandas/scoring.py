import bisect

from .auction import Contract, DeclaredContract
from .direction import Direction, Vuln


def declarer_vulnerable(declarer, vuln) -> bool:
    """
    Return whether *declarer* is vulnerable.

    *declarer* is a Direction or W/N/E/S string.
    *vuln* is a Vuln object or any string accepted by Vuln() (-, e, n, b, ew, ns, both, …).
    """
    d = Direction(declarer)
    v = Vuln(vuln)
    return v.ns_vul() if d.is_ns() else v.ew_vul()


def scorediff_imps(diff: int) -> int:
    """Convert (my_score - their_score) to IMPs."""
    imps_table = [
        15, 45, 85, 125, 165, 215, 265, 315, 365,
        425, 495, 595, 745, 895, 1095, 1295, 1495, 1745, 1995,
        2245, 2495, 2995, 3495, 3995,
    ]
    if diff < 0:
        return -bisect.bisect_left(imps_table, -diff)
    return bisect.bisect_left(imps_table, diff)


def scorediff_matchpoints(diff: int) -> float:
    """Convert (my_score - their_score) to matchpoints on a 0/0.5/1 scale."""
    if diff < 0:
        return 0.0
    if diff > 0:
        return 1.0
    return 0.5


def score_ns(declared_contract: str|DeclaredContract, tricks: int,
             vulnerable: str|Vuln) -> int:
    dc = DeclaredContract(declared_contract)
    vul = Vuln(vulnerable)
    dec_score = score(dc, tricks, vul.is_vul(dc.declarer))
    if dc.declarer.is_ew():
        return -dec_score
    else:
        return dec_score

def score(contract, tricks: int, vulnerable: bool) -> int:
    """
    Return the declarer's score for making *tricks* tricks in *contract*.

    *contract* is a Contract, DeclaredContract, Bid, or string like "3Nx".
    *tricks* is the total tricks taken (0–13).
    *vulnerable* is a bool.
    """
    con = Contract(contract)

    if tricks < con.tricks_needed:
        shortfall = con.tricks_needed - tricks
        if vulnerable:
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
        bonus = 2000 if vulnerable else 1300
    elif con.level == 6:
        bonus = 1250 if vulnerable else 800
    elif btl >= 100:
        bonus = 500 if vulnerable else 300
    else:
        bonus = 50

    bonus += 50 * con.double_state  # insult bonus

    overtricks = tricks - con.tricks_needed
    if con.double_state > 0:
        bonus += overtricks * con.double_state * (200 if vulnerable else 100)
    elif con.strain in "CcDd":
        bonus += overtricks * 20
    else:
        bonus += overtricks * 30

    return btl + bonus


__all__ = [
    "declarer_vulnerable",
    "score",
    "scorediff_imps",
    "scorediff_matchpoints",
]
