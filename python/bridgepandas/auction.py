import re
from functools import total_ordering

from .direction import Direction


@total_ordering
class Contract:
    """ Represents a contract; fields are available:
    *level* the level of the contract
    *tricks_needed* number of tricks to "make"
    *strain* one of C,D,H,S,N
    *double_state* 0 for undoubled, 1 for doubled, 2 for redoubled.
    """

    _RE = re.compile(r"([1-7])([CcDdHhSsNn]|NT|nt)([xX*]{0,2})$")

    def __init__(self, spec):
        if isinstance(spec, (Contract, DeclaredContract)):
            self.level = spec.level
            self.tricks_needed = spec.tricks_needed
            self.strain = spec.strain
            self.double_state = spec.double_state
            return
        elif isinstance(spec, Bid):
            self.level = spec.level
            self.strain = spec.strain
            self.tricks_needed = 6 + self.level
            self.double_state = 0
            return
        elif not isinstance(spec, str):
            raise TypeError("str, Bid, Contract, or DeclaredContract required")

        mo = Contract._RE.match(spec)
        if not mo:
            raise ValueError(f"Bad contract {spec!r}")

        self.level = int(mo.group(1))
        self.tricks_needed = 6 + self.level
        self.strain = mo.group(2)[0].upper()
        self.double_state = len(mo.group(3))

    def __repr__(self):
        return f"{self.level}{self.strain}{'xx'[:self.double_state]}"

    __str__ = __repr__

    def __eq__(self, other):
        o = Contract(other)
        return (self.level == o.level and self.strain == o.strain
                and self.double_state == o.double_state)

    def __lt__(self, other):
        o = Contract(other)
        if self.level != o.level:
            return self.level < o.level
        ss = "X" if self.strain == "N" else self.strain
        os = "X" if o.strain == "N" else o.strain
        if ss != os:
            return ss < os
        return self.double_state < o.double_state

    def __hash__(self):
        return hash((self.level, self.strain, self.double_state))


@total_ordering
class DeclaredContract:
    """ Represents a contract with a declarer; fields are available:
    *level* the level of the contract
    *tricks_needed* number of tricks to "make"
    *strain* one of C,D,H,S,N
    *double_state* 0 for undoubled, 1 for doubled, 2 for redoubled.
    *declarer* a Direction indicating who is declarer.
    """
    _RE = re.compile(r"([1-7])([CcDdHhSsNn]|NT|nt)([x*]{0,2})-([WNES])$")

    def __init__(self, *args):
        if len(args) == 4:
            level, strain, double_state, declarer = args
            strain = strain.upper()
            if not (1 <= level <= 7):
                raise ValueError("level must be 1–7")
            if not (0 <= double_state <= 2):
                raise ValueError("double_state must be 0, 1, or 2")
            if strain not in ("C", "D", "H", "S", "N", "NT"):
                raise ValueError(f"bad strain {strain!r}")
            self.level = level
            self.tricks_needed = 6 + level
            self.strain = strain[0]
            self.double_state = double_state
            self.declarer = Direction(declarer)
            return

        if len(args) != 1:
            raise TypeError("DeclaredContract takes 1 or 4 arguments")

        spec = args[0]
        if isinstance(spec, DeclaredContract):
            self.level = spec.level
            self.tricks_needed = spec.tricks_needed
            self.strain = spec.strain
            self.double_state = spec.double_state
            self.declarer = spec.declarer
            return

        mo = DeclaredContract._RE.match(spec)
        if not mo:
            raise ValueError(f"Bad declared contract {spec!r}")

        self.level = int(mo.group(1))
        self.tricks_needed = 6 + self.level
        self.strain = mo.group(2)[0].upper()
        self.double_state = len(mo.group(3))
        self.declarer = Direction(mo.group(4))

    def __repr__(self):
        return f"{self.level}{self.strain}{'xx'[:self.double_state]}-{self.declarer}"

    __str__ = __repr__

    def __eq__(self, other):
        o = DeclaredContract(other)
        return (self.level == o.level and self.strain == o.strain
                and self.double_state == o.double_state
                and self.declarer == o.declarer)

    def __lt__(self, other):
        o = DeclaredContract(other)
        if self.level != o.level:
            return self.level < o.level
        ss = "X" if self.strain == "N" else self.strain
        os = "X" if o.strain == "N" else o.strain
        if ss != os:
            return ss < os
        if self.double_state != o.double_state:
            return self.double_state < o.double_state
        return self.declarer.i < o.declarer.i

    def __hash__(self):
        return hash((self.level, self.strain, self.double_state, self.declarer))

    def ds(self) -> str:
        """Two-character declarer+strain string, e.g. 'NS'."""
        return str(self.declarer) + self.strain


@total_ordering
class Bid:
    """ A class representing a Bid; a level and a strain.
    *level* an ``int`` from 1 to 7
    *strain* one of C, D, H, S, N.
    Simple arithmateic is supported: Bid("3D") + 1 == Bid("3H") also
    Bid("4D") - Bid("3NT") == 2
    """

    STRAINS = ["C", "D", "H", "S", "N"]

    def __init__(self, *args):
        if len(args) == 1 and isinstance(args[0], str):
            self.level = int(args[0][0])
            self.strain = args[0][1:].upper()
        elif len(args) == 1 and isinstance(args[0], Bid):
            self.level = args[0].level
            self.strain = args[0].strain
        elif len(args) == 1 and isinstance(args[0], Call):
            if args[0].is_bid():
                self.level = args[0].bid.level
                self.strain = args[0].bid.strain
            else:
                raise TypeError(f"Cannot create Bid from non-bid Call {args[0]!r}")
        elif len(args) == 2:
            self.level = int(args[0])
            self.strain = args[1].upper()
        else:
            raise TypeError(f"Bad Bid arguments: {args!r}")

        if self.strain == "NT":
            self.strain = "N"
        if not (1 <= self.level <= 7):
            raise ValueError(f"Bad level {self.level}")
        if self.strain not in Bid.STRAINS:
            raise ValueError(f"Bad strain {self.strain!r}")

    def _step(self) -> int:
        return self.level * 5 + Bid.STRAINS.index(self.strain) - 5

    def cmp(self, other) -> int:
        if other is None:
            return 1
        if isinstance(other, Call):
            if other.is_bid():
                other = other.bid
            else:
                return 1
        return self._step() - Bid(other)._step()

    def min_bid_strain(self, strain: str) -> "Bid":
        """Return the lowest bid at or above self in the given strain."""
        if strain == "NT":
            strain = "N"
        if strain not in "CDHSN":
            raise ValueError(f"Bad strain {strain!r}")
        bid = self
        while True:
            bid += 1
            if bid.strain == strain:
                return bid

    @staticmethod
    def all_bids():
        """ Generator returning all the bids from 1C to 7N """
        cur = Bid("1C")
        top = Bid("7N")
        while True:
            yield cur
            if cur == top:
                break
            cur = cur + 1

    def all_eq_above(self):
        """ Generator returning all the bids starting from this one up to
        7NT """
        yield self
        yield from self.all_above()

    def all_above(self):
        """ Generator returning all the bids strictly higher than this one """
        cur = self
        top = Bid("7N")
        while cur < top:
            cur = cur + 1
            yield cur

    def __lt__(self, other):  return self.cmp(other) < 0
    def __le__(self, other):  return self.cmp(other) <= 0
    def __gt__(self, other):  return self.cmp(other) > 0
    def __ge__(self, other):  return self.cmp(other) >= 0
    def __eq__(self, other):  return self.cmp(other) == 0
    def __ne__(self, other):  return self.cmp(other) != 0
    def __hash__(self):       return hash((self.level, self.strain))

    def __add__(self, other):
        if not isinstance(other, int):
            raise TypeError()
        n = self._step() + other
        if not (0 <= n < 35):
            raise ValueError("Out of bounds")
        return Bid(n // 5 + 1, Bid.STRAINS[n % 5])

    def __sub__(self, other):
        if isinstance(other, int):
            n = self._step() - other
            if not (0 <= n < 35):
                raise ValueError("Out of bounds")
            return Bid(n // 5 + 1, Bid.STRAINS[n % 5])
        elif isinstance(other, Bid):
            return self._step() - other._step()
        raise TypeError()

    def __str__(self):
        return f"{self.level}{self.strain}"

    __repr__ = __str__


@total_ordering
class Call:
    """A legal auction call: bid, Pass, Double, or Redouble."""

    def __init__(self, value):
        if isinstance(value, Call):
            self.kind = value.kind
            self.bid = value.bid
        elif isinstance(value, Bid):
            self.kind = "B"
            self.bid = value
        elif isinstance(value, str):
            self.bid = None
            up = value.upper()
            if up in ("P", "PASS"):
                self.kind = "P"
            elif up in ("X", "D", "DBL", "DOUBLE"):
                self.kind = "D"
            elif up in ("XX", "R", "RDBL", "REDBL", "REDOUBLE"):
                self.kind = "R"
            else:
                self.kind = "B"
                self.bid = Bid(value)
        else:
            raise TypeError(f"Cannot initialise Call from {value!r}")

    def is_bid(self) -> bool:
        return self.kind == "B"

    def is_pass(self) -> bool:
        return self.kind == "P"

    def __eq__(self, other):
        if not isinstance(other, Call):
            other = Call(other)
        if self.is_bid():
            return self.kind == other.kind and self.bid == other.bid
        return self.kind == other.kind

    def __lt__(self, other):
        KIX = "PDRB"
        if not isinstance(other, Call):
            other = Call(other)
        if self.is_bid() and other.is_bid():
            return self.bid < other.bid
        return KIX.index(self.kind) < KIX.index(other.kind)

    def __hash__(self):
        return hash((self.kind, self.bid))

    def __str__(self):
        return str(self.bid) if self.kind == "B" else self.kind

    __repr__ = __str__


Call.PASS     = Call("P")
Call.DOUBLE   = Call("D")
Call.REDOUBLE = Call("R")


class Auction:
    """A sequence of calls in a bridge auction.

    Supports standard sequence operations: ``len(auction)`` returns the number
    of calls made, ``auction[i]`` returns the *i*-th ``Call``, and iterating
    over an auction yields each ``Call`` in order.
    """

    def __init__(self, dealer, bids=None):
        """
        Create a new Auction.  *bids* is an optional comma-separated call string.
        """
        self.dealer = dealer
        self._first_strain_calls = {}
        self._all_calls = []
        self._history = []

        self._next_dir = Direction(dealer)
        self._last_bid = None
        self._last_bid_dir = None
        self._num_doubles = 0
        self._num_passes = 0

        if bids:
            for call in bids.split(","):
                self.all(call)

    def __iter__(self):
        return iter(self._all_calls)

    def __len__(self):
        return len(self._all_calls)

    def __getitem__(self, idx):
        return self._all_calls[idx]

    def __str__(self):
        return str(self.dealer) + ":" + ",".join(map(str, self._all_calls))

    def clone(self):
        out = Auction(self.dealer)
        for call in self._all_calls:
            out.all(call)
        return out

    def turn(self) -> Direction:
        """Which direction is next to call."""
        return self._next_dir

    def done(self) -> bool:
        """ Whether the auction is legally over """
        if self._last_bid is None and self._num_passes == 4:
            return True
        if self._last_bid is not None and self._num_passes == 3:
            return True
        return False

    def final_contract(self):
        """Return the DeclaredContract, or None for a passout.  If the auction
        is not yet finished, returns what the contract will be assuming
        all players Pass from hereon."""
        if self._last_bid is None:
            return None
        key = (self._last_bid_dir.side_index(), self._last_bid.strain)
        dec, _bid = self._first_strain_calls[key]
        return DeclaredContract(self._last_bid.level, self._last_bid.strain,
                                self._num_doubles, dec)

    contract = final_contract

    def add(self, call):
        """ Add the next player's call """
        call = Call(call)
        lc = self.legal_calls()
        if lc is None:
            raise ValueError("Auction is already over")
        if call not in lc:
            raise ValueError(f"Call {call!r} is not legal here")

        self._all_calls.append(call)
        self._history.append((self._last_bid, self._last_bid_dir,
                             self._num_doubles, self._num_passes))

        cur_dir = self._next_dir
        self._next_dir += 1

        if call.is_pass():
            self._num_passes += 1
            return self

        if call == Call.DOUBLE:
            self._num_passes = 0
            self._num_doubles = 1
            return self

        if call == Call.REDOUBLE:
            self._num_passes = 0
            self._num_doubles = 2
            return self

        assert call.is_bid()
        bid = call.bid
        key = (cur_dir.side_index(), bid.strain)
        if key not in self._first_strain_calls:
            self._first_strain_calls[key] = (cur_dir, bid)
        self._last_bid = bid
        self._last_bid_dir = cur_dir
        self._num_doubles = 0
        self._num_passes = 0
        return self

    def undo(self):
        """ Undo the previous player's call """
        if not self._all_calls:
            raise ValueError("No calls to undo")
        self._next_dir -= 1
        off = self._all_calls.pop()
        self._last_bid, self._last_bid_dir, self._num_doubles, self._num_passes = self._history.pop()
        if off.is_bid():
            key = (self._next_dir.side_index(), off.bid.strain)
            fdir, fbid = self._first_strain_calls[key]
            if off.bid == fbid:
                del self._first_strain_calls[key]
        return self

    def legal_calls(self):
        """Return list of legal calls, or None if the auction is over."""
        if self.done():
            return None
        out = [Call.PASS]
        if self._last_bid is None:
            out.extend(map(Call, Bid.all_bids()))
        else:
            out.extend(map(Call, self._last_bid.all_above()))
            if self._num_doubles == 0 and self._next_dir.opp_side(self._last_bid_dir):
                out.append(Call.DOUBLE)
            elif self._num_doubles == 1 and self._next_dir.same_side(self._last_bid_dir):
                out.append(Call.REDOUBLE)
        return out

    def min_bid(self):
        """Lowest legal bid, or None if already at 7NT."""
        if self._last_bid is None:
            return Bid("1C")
        if self._last_bid.level == 7 and self._last_bid.strain == "N":
            return None
        return self._last_bid + 1


__all__ = [
    "Auction",
    "Bid",
    "Call",
    "Contract",
    "DeclaredContract",
]
