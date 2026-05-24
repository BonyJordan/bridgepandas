class Direction:
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
        return (self.i & 1) == (Direction(other).i & 1)

    def opp_side(self, other) -> bool:
        return (self.i & 1) != (Direction(other).i & 1)

    def dir_pair(self) -> str:
        return ("EW", "NS", "EW", "NS")[self.i]

    def side_index(self) -> int:
        return self.i & 1

    def is_ew(self) -> bool:
        return self.i & 1 == 0

    def is_ns(self) -> bool:
        return self.i & 1 == 1

    @staticmethod
    def all_dirs() -> list:
        return [Direction(i) for i in range(4)]


Direction.WEST  = Direction("W")
Direction.NORTH = Direction("N")
Direction.EAST  = Direction("E")
Direction.SOUTH = Direction("S")


class Vuln:
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
                raise ValueError("Vuln int must be 0–3")
            self.data = data
        elif isinstance(data, str):
            lc = data.lower()
            if lc not in Vuln._TABLE:
                raise ValueError(f"Unknown vulnerability {data!r}; use -, e, n, b, ew, ns, both")
            self.data = Vuln._TABLE[lc]
        elif isinstance(data, Vuln):
            self.data = data.data
        else:
            raise TypeError(f"Cannot construct Vuln from {type(data)}")

    def ew_vul(self) -> bool:
        return bool(self.data & Vuln.EW_BIT)

    def ns_vul(self) -> bool:
        return bool(self.data & Vuln.NS_BIT)

    def is_vul(self, direction) -> bool:
        d = Direction(direction)
        return self.ns_vul() if d.is_ns() else self.ew_vul()

    def __str__(self):
        return "-neb"[self.data]

    def __repr__(self):
        return f"Vuln({str(self)!r})"

    def __eq__(self, other):
        return isinstance(other, Vuln) and self.data == other.data

    def __hash__(self):
        return hash(self.data)

    @staticmethod
    def all_vulns() -> list:
        return [Vuln(i) for i in range(4)]


def board_number_to_dealer_vuln(num: int):
    """Return (dealer: Direction, vuln: Vuln) for a board number (1-based)."""
    n = (num + 15) % 16
    d = n % 4
    v = (d + (n // 4)) % 4
    return Direction.NORTH + d, Vuln("-neb"[v])


def dealer_vuln_to_board_number(dealer, vuln) -> int:
    """Return the board number (1-based) for a given dealer and vulnerability."""
    dix = (Direction(dealer).i - Direction.NORTH.i) & 3
    vix = Vuln(vuln).data
    delta = (vix + 4 - dix) % 4
    return 1 + delta * 4 + dix


__all__ = [
    "Direction",
    "Vuln",
    "board_number_to_dealer_vuln",
    "dealer_vuln_to_board_number",
]
