# bridgepandas

A Python library for bridge deal generation, hand analysis, and double-dummy simulation, built on top of pandas.

## Features

- **Deal generation** — fast random deals via NumPy shuffles, BDD-constrained sampling, or accept/reject filtering
- **Hand analysis** — HCP, suit lengths, losers, controls, quick tricks, shape, and more, as pandas-native array accessors
- **Double-dummy solving** — batch solver via [DDS](https://github.com/dds-bridge/dds) with per-declarer caching and tqdm progress bars
- **Scoring** — duplicate scoring, IMP and matchpoint differentials, per-board vulnerability
- **Auction tools** — `Auction`, `Contract`, `DeclaredContract`, bid arithmetic

## Quick start

```python
import bridgepandas as bp

# Generate random deals
df = bp.random_deals(1000, seed=42)

# Hand properties as pandas Series
df['north'].hcp
df['south'].spades

# Constrain with a HandSet
hs = (bp.h.SPADES == 5) & (bp.h.HCP >= 15) & (bp.h.HCP <= 17)
df = bp.random_deals(100, south=hs, seed=1)

# Accept/reject filter (combined with HandSet for speed)
def accept(deal):
    return deal.south.handshape in {(5,3,3,2), (4,4,3,2)}

df = bp.random_deals(500, south=hs, accept=accept, seed=1)

# Double-dummy scoring
bp.add_dds_score(df, '3N-S', 'score_3nt', '-')   # neither vulnerable
bp.add_dds_score(df, '4S-S', 'score_4s',  'ns')  # NS vulnerable

# IMP differential
df['imps'] = bp.scorediff_imps(df['score_3nt'] - df['score_4s'])
```

## Running the tests

The unit tests cover hand encoding, deal generation, and hand-set constraints.
They require only `numpy`, `pandas`, and `pytest` — no DDS solver needed.

```bash
source venv/bin/activate
pip install pytest            # if not already installed
pytest python/tests/ -v
```

Individual test files:

```bash
pytest python/tests/test_hand.py     -v   # hand encoding, HCP, shape, accessors
pytest python/tests/test_deal.py     -v   # deal generation and structure
pytest python/tests/test_handset.py  -v   # HandSet constraint logic
```

## Examples

See [`examples/`](examples/) for Jupyter notebooks:

- **`intro.ipynb`** — walkthrough of core features
- **`pancake_stayman.ipynb`** — double-dummy study: does Stayman gain or lose on flat hands?

## Installation

```
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate

# Download and compile DDS3.  Technically optional, but
# you won't have much fun simulating without it.
git clone https://github.com/dds-bridge/dds.git
cd dds
bazel build //python:dds3_wheel_dist
pip install bazel-bin/python/dds3-*-py3-none-any.whl
cd ..

# Install other useful packages
pip install numpy         # required
pip install pandas        # required
pip install scipy         # not required, but needed for example notebooks
pip install matplotlib    # not required, but needed for example notebooks
pip install tqdm          # not required, but provides progress bars

# Install our package
pip install -e python/

```
