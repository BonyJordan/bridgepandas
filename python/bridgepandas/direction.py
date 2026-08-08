import numpy as np
import pandas as pd


class Direction:
    """ A direction around the bridge table. You can initialize from a
    ``str``, one of W,N,E,S, or an ``int`` (west=0), or use the built in
    directional constants like ``Direction.WEST``.

    You can also say e.g.  ``direction + 1`` to get the next direction 
    around the table."""


    ALL = "WNES"  # W=0, N=1, E=2, S=3 — clockwise, even=EW, odd=NS

    def __init__(self, name):
        if isinstance(name, str):
            self.i = Direction.ALL.index(name)
        elif isinstance(name, int):
            self.i = name
        elif isinstance(name, Direction):
            self.i = name.i
        else:
            raise TypeError(f"Cannot construct Direction from {type(name)}")

    def __add__(self, other):
        if isinstance(other, int):
            return Direction((self.i + other) & 3)
        raise TypeError()

    def __sub__(self, other):
        if isinstance(other, int):
            return Direction((self.i - other) & 3)
        elif isinstance(other, Direction):
            return (self.i - other.i) & 3
        raise TypeError()

    def __str__(self):
        return Direction.ALL[self.i]

    def __repr__(self):
        return f"Direction({Direction.ALL[self.i]!r})"

    def __eq__(self, other):
        return self.i == Direction(other).i

    def __hash__(self):
        return hash(self.i)

    def same_side(self, other) -> bool:
        """ Are these two directions partners (or the same player) """
        return (self.i & 1) == (Direction(other).i & 1)

    def opp_side(self, other) -> bool:
        """ Are these two directions opponents? """
        return (self.i & 1) != (Direction(other).i & 1)

    def dir_pair(self) -> str:
        """ Return either "EW" or "NS" """
        return ("EW", "NS", "EW", "NS")[self.i]

    def side_index(self) -> int:
        """ Return the direction as an int.  West=0 """
        return self.i & 1

    def is_ew(self) -> bool:
        return self.i & 1 == 0

    def is_ns(self) -> bool:
        return self.i & 1 == 1

    @staticmethod
    def all_dirs() -> list:
        """ Static method returning a list of all directions """
        return [Direction(i) for i in range(4)]


Direction.WEST  = Direction("W")
Direction.NORTH = Direction("N")
Direction.EAST  = Direction("E")
Direction.SOUTH = Direction("S")


class TableVuln:
    EW_BIT = 2
    NS_BIT = 1

    _TABLE = {
        "-": 0, "o": 0, "none": 0,
        "e": EW_BIT, "ew": EW_BIT,
        "n": NS_BIT, "ns": NS_BIT,
        "b": EW_BIT | NS_BIT, "both": EW_BIT | NS_BIT, "all": EW_BIT | NS_BIT,
    }

    def __init__(self, data):
        if isinstance(data, int):
            if data < 0 or data > 3:
                raise ValueError("TableVuln int must be 0–3")
            self.data = data
        elif isinstance(data, str):
            lc = data.lower()
            if lc not in TableVuln._TABLE:
                raise ValueError(f"Unknown vulnerability {data!r}; use -, e, n, b, ew, ns, both")
            self.data = TableVuln._TABLE[lc]
        elif isinstance(data, TableVuln):
            self.data = data.data
        else:
            raise TypeError(f"Cannot construct TableVuln from {type(data)}")

    def ew_vul(self) -> bool:
        return bool(self.data & TableVuln.EW_BIT)

    def ns_vul(self) -> bool:
        return bool(self.data & TableVuln.NS_BIT)

    def is_vul(self, direction) -> bool:
        d = Direction(direction)
        return self.ns_vul() if d.is_ns() else self.ew_vul()

    def __str__(self):
        return "-neb"[self.data]

    def __repr__(self):
        return f"TableVuln({str(self)!r})"

    def __eq__(self, other):
        return isinstance(other, TableVuln) and self.data == other.data

    def __hash__(self):
        return hash(self.data)

    @staticmethod
    def all_vulns() -> list:
        return [TableVuln(i) for i in range(4)]


def _factorize_attrs(values, ctor, **extractors):
    """
    Construct ``ctor(v)`` once per distinct value in *values*, pull one or
    more plain-Python attributes out of each unique object via *extractors*
    (name -> callable(obj) -> value), and broadcast the results back out to
    arrays aligned with *values*.

    This lets Series-taking functions reuse a class's own scalar
    parsing/construction logic (``Direction(...)``, ``TableVuln(...)``,
    ``Contract(...)``) as the single source of truth, while only doing the
    (possibly regex-based) construction once per unique value rather than
    once per row.

    Returns a dict *name -> np.ndarray*, one entry per keyword in
    *extractors*, each the same length as *values*.
    """
    codes, uniques = pd.factorize(values)
    objs = [ctor(u) for u in uniques]
    return {
        name: np.array([fn(o) for o in objs])[codes]
        for name, fn in extractors.items()
    }


def board_number_to_dealer_vuln(num):
    """Return the tuple (dealer: Direction, vuln: TableVuln) for a board number (1-based).

    *num* may instead be a pandas Series of board numbers, in which case the
    result is a tuple of two pandas Series (dealer, vuln) of the same length,
    preserving *num*'s Index.
    """
    if not isinstance(num, pd.Series):
        n = (num + 15) % 16
        d = n % 4
        v = (d + (n // 4)) % 4
        return Direction.NORTH + d, TableVuln("-neb"[v])

    n = (num.to_numpy() + 15) % 16
    d = n % 4
    v = (d + (n // 4)) % 4

    dealers = np.array([Direction.NORTH + i for i in range(4)], dtype=object)
    vulns = np.array([TableVuln(i) for i in range(4)], dtype=object)

    return (pd.Series(dealers[d], index=num.index),
            pd.Series(vulns[v], index=num.index))


def dealer_vuln_to_board_number(dealer, vuln) -> int:
    """Return the board number (1-based) for a given dealer and vulnerability.

    Either *dealer* or *vuln* may instead be a pandas Series, in which case
    the result is a pandas Series of int with the same length (the Index is
    taken from *dealer* if it is a Series, else from *vuln*).
    """
    dealer_is_series = isinstance(dealer, pd.Series)
    vuln_is_series = isinstance(vuln, pd.Series)

    if not dealer_is_series and not vuln_is_series:
        dix = (Direction(dealer).i - Direction.NORTH.i) & 3
        vix = TableVuln(vuln).data
        delta = (vix + 4 - dix) % 4
        return 1 + delta * 4 + dix

    if dealer_is_series:
        dix = (_factorize_attrs(dealer, Direction, i=lambda d: d.i)["i"] - Direction.NORTH.i) & 3
    else:
        dix = (Direction(dealer).i - Direction.NORTH.i) & 3

    if vuln_is_series:
        vix = _factorize_attrs(vuln, TableVuln, data=lambda v: v.data)["data"]
    else:
        vix = TableVuln(vuln).data

    delta = (vix + 4 - dix) % 4
    result = 1 + delta * 4 + dix

    index = dealer.index if dealer_is_series else vuln.index
    return pd.Series(result, index=index)


__all__ = [
    "Direction",
    "TableVuln",
    "board_number_to_dealer_vuln",
    "dealer_vuln_to_board_number",
]
