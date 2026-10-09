# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`bridgepandas` is a Python library for bridge deal generation, hand analysis, and double-dummy
simulation, built on top of pandas. The package lives in `python/bridgepandas/`; everything else
(`docs/`, `examples/`, `studies/`) is documentation, notebooks, or one-off analysis scripts that
import the installed package.

## Environment

Activate the project venv before running any Python (`source ~/ve/bpan/bin/activate`) — the
system Python lacks pandas/numpy. The package includes a compiled C++ extension
(`bridgepandas.jbdd`), so after editing `python/jbdd/*.cpp`/`*.h` you must rebuild:

```
pip install -e python/
```

## Common commands

```bash
# Run the full test suite (from repo root or python/)
cd python && python -m pytest tests/

# Run a single test file / test
python -m pytest python/tests/test_hand.py
python -m pytest python/tests/test_hand.py::test_hcp -v

# Rebuild the C++ extension after touching jbdd sources
pip install -e python/

# Regenerate docs/reference.html after changing public API or docstrings
python build_reference.py
```

DDS (double-dummy solving) requires the separate `dds3` wheel built from
https://github.com/dds-bridge/dds (see README.md for the bazel build steps). It's optional —
everything except `bp.solve`/`bp.add_dds_score`/`bp.add_dds_tricks` works without it.

## Architecture

### Hand representation (`hand.py`)

Every hand is a 52-bit integer: bit `suit_offset + rank_index` is set for each held card.
Suit offsets are `C=0, D=13, H=26, S=39`; rank indices are `2=0 … T=8, J=9, Q=10, K=11, A=12`.
This packed-int representation is the foundation everything else builds on:

- **`Hand`** — an `int` subclass for a single hand. Scalar properties (`.hcp`, `.handshape`,
  `.losers`, `.quick_tricks`, …) mirror the vectorized Series accessors below.
- **`BridgeHandDtype` / `BridgeHandArray`** — a pandas `ExtensionDtype`/`ExtensionArray` pair that
  stores a column of hands as a single `int64` numpy array (plus a bool NA mask). This is what
  makes `df['north']` a real pandas column.
- **Series accessors** — `hcp`, `spades`/`hearts`/`diamonds`/`clubs`, `handshape`, `pattern`,
  `losers`, `quick_tricks`, `controls`, `num(spec)`, `suits_of(spec)`, `has(card)`,
  `good_suit(spec, suit)`, `match_shape(spec)`, etc. are registered via
  `pd.api.extensions.register_series_accessor` directly on `BridgeHandArray` columns (e.g.
  `df.north.hcp`). Most are implemented as **vectorized bit tricks over the raw `int64` array**
  (lookup tables like `_POPCOUNT13`, masked shifts/adds) rather than per-row Python loops — when
  adding a new metric, follow that pattern instead of using `.apply()`.
- A scalar `Hand` property and its Series-accessor counterpart should always agree — they're
  meant to be two views of the same bit math, so changes to one almost always need the other.

### Deal generation (`deal.py`)

`random_deals()` is the main entry point and dispatches to one of three strategies depending on
the constraints given for each seat (`west=`, `north=`, `east=`, `south=`) and whether an
`accept=` post-filter is given:

1. **Pure numpy shuffle** (no constraints at all) — fastest path, just permutes a 52-card deck.
2. **BDD sampling** (`handset.py`) — used when constraints are `None`/str/int/`HandSet` (no plain
   callables); converts each seat's constraint into a `HandSet`, combines them into a `DealSet`,
   and samples directly from the constrained deal space.
3. **Accept/reject** — used when any constraint is a plain callable; deals cards directly and
   retries until constraints are satisfied. Can be slow for rare combinations of constraints
   (governed by `fail_count`).

`Deal` is an immutable, hashable value object (four `Hand`s) usable as a dict key/set member, and
constructible from a DataFrame row (`Deal.from_row`) or hand strings.

### Constraint engine (`handset.py`)

This is the most intricate module — a small BDD (Binary Decision Diagram) library wrapper used to
represent and exactly sample from large sets of hands/deals without enumerating them.

- The BDD engine itself is the C++ extension `jbdd` (`python/jbdd/jbdd.cpp`, `j128.cpp`) exposed
  to Python as `bridgepandas.jbdd.BDD`.
- **BDD variable ordering** matters and is documented at the top of the file: for a `HandSet`
  (single hand, 52 vars) it's honors interleaved by suit, then low cards by suit; for a `DealSet`
  (full deal, 104 vars) it's 2 variables per card encoding which of the 4 players holds it.
- **`HandSetMetric`** (and its `SimpleHandMetric`/`QuickTricksMetric`/`LosersMetric` subclasses)
  represent an integer-valued function over hands as a `dict[value -> BDD]`, supporting
  `+`, `-`, `*`, and comparisons (`<`, `<=`, `>=`, `>`, `==`) that produce a `HandSet`.
- **`HandSet`** wraps a BDD for one hand; **`DealSet`** wraps a BDD over all 104 deal variables.
  `DealSetConverter` (`hand_makers.WEST/NORTH/EAST/SOUTH`) lifts a per-seat `HandSet` into a
  `DealSet` by re-encoding each single-hand variable as the matching pair of deal variables.
- **`hand_makers`** (aliased as `h`, and re-exported from the package as `bp.h`) is the public
  surface: lazily-computed class attributes (`h.HCP`, `h.SPADES`, `h.LOSERS`, …) and static
  methods (`h.MATCH_SHAPE(spec)`, `h.GOOD_SUIT(spec, suit)`, `h.HAS(card)`, `h.NUM(spec)`) that
  combine with `&`, `|`, `~`, and comparison operators to build `HandSet`/`DealSet` constraints,
  e.g. `(h.HCP >= 15) & (h.HCP <= 17) & h.MATCH_SHAPE("any 4333 + any 4432")`.
- Shape spec syntax (`"4432"`, `"any 4432"`, `"44xx"`, `"4432 + 4333"`, `"44xx - 4450"`) is parsed
  once in `shape.py` (no BDD dependency, importable from both `hand.py` and `handset.py` without
  circularity) and consumed both by the scalar `Hand.match_shape()` and by `ShapeMaker` here.

### Double-dummy solving (`dds.py`)

Wraps the external `dds3` package (a ctypes binding to the DDS C library) to batch-solve deals.
Hands are converted to PBN strings; boards are solved in chunks of `_MAXNOOFBOARDS` (200, a DDS
library limit), optionally across multiple processes (`processes=`). `add_dds_score`/
`add_dds_tricks` cache results per-deal in a `_dds` dict column keyed by `declarer+strain` (e.g.
`"NH"`), since trick count depends only on trump and leader, not on level/doubling — so scoring
the same contract twice, or scoring multiple contracts that share a declarer+strain, reuses cached
solves.

### Auction / scoring (`auction.py`, `scoring.py`, `direction.py`)

`Direction` (W/N/E/S, int-indexed, supports `+`/`-` arithmetic around the table) and `TableVuln`
are small value types used throughout. `Contract`/`DeclaredContract`/`Bid`/`Call`/`Auction` model
bidding and contracts; `scoring.py` turns a `DeclaredContract` + trick count + vulnerability into a
duplicate score, and provides `scorediff_imps`/`scorediff_matchpoints` for converting score
differences into IMPs/matchpoints.

## Docs

`docs/reference.html` is a generated single-page API reference, built by `build_reference.py` from
the installed package's docstrings (`bp.__pdoc__` hides internal names). Regenerate it after
changing any public docstring or adding/removing public API, and commit the regenerated file
alongside the source change (see recent commits for the pattern).
