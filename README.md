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
    return deal.south.shape in {(5,3,3,2), (4,4,3,2)}

df = bp.random_deals(500, south=hs, accept=accept, seed=1)

# Double-dummy scoring
bp.add_dds(df, '3N-S', 'score_3nt', '-')   # neither vulnerable
bp.add_dds(df, '4S-S', 'score_4s',  'ns')  # NS vulnerable

# IMP differential
df['imps'] = (df['score_3nt'] - df['score_4s']).map(bp.scorediff_imps)
```

## Examples

See [`examples/`](examples/) for Jupyter notebooks:

- **`intro.ipynb`** — walkthrough of core features
- **`pancake_stayman.ipynb`** — double-dummy study: does Stayman gain or lose on flat hands?

## Installation

```bash
pip install -e python/
```

The double-dummy solver requires the `dds3` package (wraps the DDS library).
