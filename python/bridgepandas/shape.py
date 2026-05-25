"""
Shape spec parsing: converts strings like "any 5332 + 4441 - 4450" into a
frozenset of (spades, hearts, diamonds, clubs) length tuples.

This module contains no BDD or HandSet dependencies so it can be imported by
both hand.py and handset.py without circular imports.
"""

import functools
import itertools
import re


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tuple_to_pattern(tupe, n):
    s = [-1] + list(tupe) + [n + len(tupe)]
    out = tuple(s[i+1] - s[i] - 1 for i in range(len(tupe) + 1))
    assert min(out) >= 0
    return out


# All valid (spades, hearts, diamonds, clubs) length tuples for a 13-card hand.
_ALL_PATTERNS = [_tuple_to_pattern(t, 13) for t in itertools.combinations(range(16), 3)]

_RE = re.compile(
    r'(?P<SKIP>\s+)|(?P<ANY>any)|(?P<OP>[-+])|(?P<PAT>[0-9x]{4})|(?P<ERROR>.)'
)


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def _matching(spec):
    assert len(spec) == 4
    return [t for t in _ALL_PATTERNS if all(spec[i] in ('x', str(t[i])) for i in range(4))]


class _ExBase:
    def do_end(self): raise ValueError("Unexpected end of shape spec")
    def do_skip(self, mo, sf): return self
    def do_any(self, mo, sf): raise ValueError(f"'any' unexpected at position {mo.start()+1}")
    def do_op(self, mo, sf): raise ValueError(f"Operator unexpected at position {mo.start()+1}")
    def do_pat(self, mo, sf): raise ValueError(f"Pattern unexpected at position {mo.start()+1}")


class _ExAnyOrPat(_ExBase):
    def __init__(self, sign): self.sign = sign
    def do_any(self, mo, sf): return _ExPat(self.sign)
    def do_pat(self, mo, sf):
        (sf.update if self.sign == '+' else sf.difference_update)(_matching(mo.group()))
        return _ExOp()


class _ExPat(_ExBase):
    def __init__(self, sign): self.sign = sign
    def do_pat(self, mo, sf):
        tupes = set()
        for perm in set(itertools.permutations(mo.group())):
            tupes.update(_matching(perm))
        (sf.update if self.sign == '+' else sf.difference_update)(tupes)
        return _ExOp()


class _ExOp(_ExBase):
    def do_end(self): pass
    def do_op(self, mo, sf): return _ExAnyOrPat(mo.group())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@functools.cache
def parse_shape_spec(spec: str) -> frozenset:
    """Parse a shape spec into a frozenset of (spades, hearts, diamonds, clubs) tuples.

    Syntax examples::

        "4432"           – exactly 4S 4H 3D 2C
        "any 4432"       – any permutation of 4-4-3-2
        "44xx"           – 4S 4H, any D and C
        "4432 + 4333"    – either shape
        "44xx - 4450"    – 4-4 majors, not 4-4-5-0
    """
    state = _ExAnyOrPat("+")
    so_far: set = set()
    for mo in _RE.finditer(spec):
        grp = mo.lastgroup
        if grp == "SKIP":
            continue
        elif grp == "ANY":
            state = state.do_any(mo, so_far)
        elif grp == "OP":
            state = state.do_op(mo, so_far)
        elif grp == "PAT":
            state = state.do_pat(mo, so_far)
        else:
            raise ValueError(f"Unexpected character '{mo.group()}' at position {mo.start()+1}")
    state.do_end()
    return frozenset(so_far)
