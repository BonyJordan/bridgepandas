from __future__ import annotations

import ctypes
import importlib.util

import numpy as np
import pandas as pd

from .hand import RANKS

_MAXNOOFBOARDS = 200

# DDS encoding maps
_TRUMP_MAP: dict[str, int] = {"S": 0, "H": 1, "D": 2, "C": 3, "NT": 4, "N": 4}
_LEADER_MAP: dict[str, int] = {"N": 0, "E": 1, "S": 2, "W": 3}


# ---------------------------------------------------------------------------
# ctypes struct definitions matching dll.h
# ---------------------------------------------------------------------------

class _DealPBN(ctypes.Structure):
    _fields_ = [
        ("trump",            ctypes.c_int),
        ("first",            ctypes.c_int),
        ("currentTrickSuit", ctypes.c_int * 3),
        ("currentTrickRank", ctypes.c_int * 3),
        ("remainCards",      ctypes.c_char * 80),
    ]


class _BoardsPBN(ctypes.Structure):
    _fields_ = [
        ("no_of_boards", ctypes.c_int),
        ("deals",        _DealPBN * _MAXNOOFBOARDS),
        ("target",       ctypes.c_int * _MAXNOOFBOARDS),
        ("solutions",    ctypes.c_int * _MAXNOOFBOARDS),
        ("mode",         ctypes.c_int * _MAXNOOFBOARDS),
    ]


class _FutureTricks(ctypes.Structure):
    _fields_ = [
        ("nodes",  ctypes.c_int),
        ("cards",  ctypes.c_int),
        ("suit",   ctypes.c_int * 13),
        ("rank",   ctypes.c_int * 13),
        ("equals", ctypes.c_int * 13),
        ("score",  ctypes.c_int * 13),
    ]


class _SolvedBoards(ctypes.Structure):
    _fields_ = [
        ("no_of_boards", ctypes.c_int),
        ("solved_board", _FutureTricks * _MAXNOOFBOARDS),
    ]


_lib: ctypes.CDLL | None = None


def _get_lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    spec = importlib.util.find_spec("dds3._dds3")
    if spec is None or spec.origin is None:
        raise ImportError(
            "dds3 package not found; install it with:\n"
            "  pip install <path>/dds3-*.whl"
        )
    lib = ctypes.CDLL(spec.origin)
    lib.SolveAllBoards.argtypes = [
        ctypes.POINTER(_BoardsPBN),
        ctypes.POINTER(_SolvedBoards),
    ]
    lib.SolveAllBoards.restype = ctypes.c_int
    _lib = lib
    return _lib


# ---------------------------------------------------------------------------
# PBN string builder
# ---------------------------------------------------------------------------

_RANK_BITS: list[tuple[int, str]] = [(12 - i, ch) for i, ch in enumerate(RANKS)]


def _suit_mask_to_str(mask: int) -> str:
    return "".join(ch for bit, ch in _RANK_BITS if mask & (1 << bit))


def _hand_to_pbn(north: int, east: int, south: int, west: int) -> bytes:
    """Build the DealPBN.remainCards byte string from four 52-bit hand ints."""
    parts = []
    for hand in (north, east, south, west):
        suits = ".".join(
            _suit_mask_to_str((hand >> offset) & 0x1FFF)
            for offset in (39, 26, 13, 0)  # S, H, D, C
        )
        parts.append(suits)
    return ("N:" + " ".join(parts)).encode()


# ---------------------------------------------------------------------------
# Core batch solver (leader-side tricks)
# ---------------------------------------------------------------------------

def _solve_batch(
    df: pd.DataFrame,
    trump_arr: np.ndarray,
    leader_arr: np.ndarray,
) -> np.ndarray:
    """
    Solve all rows, returning int8 array of tricks for the *leader's* side.

    trump_arr and leader_arr are int arrays of length len(df) with per-row
    DDS trump (0-4) and DDS first (0-3) values.
    """
    lib = _get_lib()
    n_rows = len(df)
    results = np.empty(n_rows, dtype=np.int8)

    north_vals = np.asarray(df["north"], dtype=np.int64)
    east_vals  = np.asarray(df["east"],  dtype=np.int64)
    south_vals = np.asarray(df["south"], dtype=np.int64)
    west_vals  = np.asarray(df["west"],  dtype=np.int64)

    boards = _BoardsPBN()
    solved = _SolvedBoards()

    for start in range(0, n_rows, _MAXNOOFBOARDS):
        end = min(start + _MAXNOOFBOARDS, n_rows)
        count = end - start
        boards.no_of_boards = count

        for i in range(count):
            idx = start + i
            pbn = _hand_to_pbn(
                int(north_vals[idx]),
                int(east_vals[idx]),
                int(south_vals[idx]),
                int(west_vals[idx]),
            )
            d = boards.deals[i]
            d.trump = int(trump_arr[idx])
            d.first = int(leader_arr[idx])
            d.currentTrickSuit[0] = d.currentTrickSuit[1] = d.currentTrickSuit[2] = 0
            d.currentTrickRank[0] = d.currentTrickRank[1] = d.currentTrickRank[2] = 0
            d.remainCards = pbn
            boards.target[i]    = -1
            boards.solutions[i] = 1
            boards.mode[i]      = 1

        rc = lib.SolveAllBoards(ctypes.byref(boards), ctypes.byref(solved))
        if rc != 1:
            raise RuntimeError(f"SolveAllBoards returned error code {rc}")

        for i in range(count):
            results[start + i] = solved.solved_board[i].score[0]

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse_trump(trump: int | str) -> int:
    if isinstance(trump, str):
        v = _TRUMP_MAP.get(trump.upper())
        if v is None:
            raise ValueError(f"Unknown trump {trump!r}; use S, H, D, C, or NT")
        return v
    return int(trump)


def _parse_leader(leader: int | str) -> int:
    if isinstance(leader, str):
        v = _LEADER_MAP.get(leader.upper())
        if v is None:
            raise ValueError(f"Unknown leader {leader!r}; use N, E, S, or W")
        return v
    return int(leader)


def solve(
    df: pd.DataFrame,
    trump: int | str,
    leader: int | str,
) -> np.ndarray:
    """
    Compute double-dummy trick counts for every deal in *df*.

    Parameters
    ----------
    df : DataFrame
        Must have columns 'north', 'east', 'south', 'west'.
    trump : int or str
        0/S, 1/H, 2/D, 3/C, 4/NT.
    leader : int or str
        0/N, 1/E, 2/S, 3/W — direction of the opening lead.

    Returns
    -------
    numpy.ndarray of int8, shape (len(df),)
        Tricks the leader's side can take with double-dummy play.
    """
    n = len(df)
    trump_arr  = np.full(n, _parse_trump(trump),  dtype=np.int8)
    leader_arr = np.full(n, _parse_leader(leader), dtype=np.int8)
    return _solve_batch(df, trump_arr, leader_arr)


def _dds_key(dc) -> str:
    """Cache key: declarer direction + strain, e.g. 'NH', 'WN' (NT), 'SC'."""
    return str(dc.declarer) + dc.strain


def add_dds(
    df: pd.DataFrame,
    contracts,
    col_name: str,
    vuln,
) -> None:
    """
    Solve double-dummy and score each deal, adding the NS score as a new column.

    Results are cached in a ``_dds`` column (dict per row, keyed by
    declarer+strain e.g. ``"NH"``) so repeated calls for the same
    declarer/strain combination skip the solver entirely.

    Parameters
    ----------
    df : DataFrame
        Must have columns 'north', 'east', 'south', 'west'. Modified in place.
    contracts : str | DeclaredContract | pd.Series
        A single contract applied to every row, or a per-row Series.
        Strings are parsed as DeclaredContract (e.g. ``"3N-N"``, ``"4Sx-E"``).
    col_name : str
        Name of the column to add (or overwrite) in *df*.
    vuln : str | Vuln | pd.Series
        Vulnerability. A scalar is applied to every row; a Series supplies
        per-row values.
    """
    from .auction import DeclaredContract
    from .scoring import score_ns

    n = len(df)

    # Normalise contracts to a list of DeclaredContract
    if isinstance(contracts, pd.Series):
        dc_list = [DeclaredContract(c) for c in contracts]
    else:
        dc = DeclaredContract(contracts)
        dc_list = [dc] * n

    # Ensure the cache column exists
    if "_dds" not in df.columns:
        df["_dds"] = [dict() for _ in range(n)]

    # Build per-row cache keys and find rows not yet solved
    keys = [_dds_key(dc) for dc in dc_list]
    cache = df["_dds"]
    needs_solve = [i for i in range(n) if keys[i] not in cache.iat[i]]

    if needs_solve:
        # Build trump/leader arrays only for the rows that need solving
        trump_arr  = np.empty(n, dtype=np.int8)
        leader_arr = np.empty(n, dtype=np.int8)
        for i in needs_solve:
            dc = dc_list[i]
            trump_arr[i]  = _TRUMP_MAP[dc.strain]
            leader_arr[i] = _LEADER_MAP[str(dc.declarer + 1)]

        sub_df = df.iloc[needs_solve]
        leader_tricks = _solve_batch(sub_df, trump_arr[needs_solve], leader_arr[needs_solve])

        for j, i in enumerate(needs_solve):
            cache.iat[i][keys[i]] = 13 - int(leader_tricks[j])

    # Score from NS perspective using cached trick counts
    vuln_series = vuln if isinstance(vuln, pd.Series) else None
    scores = np.empty(n, dtype=np.int32)
    for i, dc in enumerate(dc_list):
        v = vuln_series.iloc[i] if vuln_series is not None else vuln
        scores[i] = score_ns(dc, cache.iat[i][keys[i]], v)

    df[col_name] = scores
