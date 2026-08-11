# Synthetic Data Methodology

*How I generate anonymized practice datasets for portfolio and blog work — and why I don't just make up plausible-looking numbers.*

## Why this exists

A lot of portfolio and blog content built on "realistic" data quietly cuts a corner: either the numbers are real (and shouldn't be public), or they're invented on the spot and don't actually hold together as a coherent dataset — bar charts where the percentages don't sum correctly, a headline stat that doesn't match a chart three paragraphs later. Both are credibility risks with a technical audience.

This repo documents the approach I use instead: generate data that is **fully synthetic** but **internally consistent** — every number in every chart is actually computed from the same underlying dataset, not typed in separately by hand. That sounds like a small distinction. It isn't. It's the difference between "these charts illustrate a plausible scenario" and "these charts don't actually agree with each other," and the second one is only visible to a reader who checks — which, on a technical platform, is exactly the reader you most want to get right.

## The core technique: percentile-anchored generation

Rather than picking a distribution family and hoping its default shape looks right, I build the dataset by specifying the target *shape* directly — a handful of percentile anchor points (e.g., "the 25th percentile should be 1, the median should be 3, the 75th percentile should be 8") — and generate individual data points via interpolation between those anchors, with Poisson noise layered on top so the result looks like organic data rather than a mathematically smooth staircase.

```python
anchor_pct = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]
anchor_val = [0,   0,    1,    3,    8,    20]
ranks = (np.arange(n) + 0.5) / N
target_curve = np.interp(ranks, anchor_pct, anchor_val)
values = rng.poisson(np.clip(target_curve, 0.05, None))
```

This guarantees the generated data's actual percentiles land close to the intended targets *by construction*, rather than by trial and error with a generic distribution's parameters.

## Controlling a specific finding: volume concentration

Some illustrative scenarios need a specific relationship to hold — for example, "a small minority of users should account for a large share of total volume." Rather than hoping a random draw happens to produce that pattern, I split the population into two segments and directly solve for the scaling factor that makes the *concentration* land exactly on target:

```python
target_sum = target_share * base_sum / (1 - target_share)
tail_vals = (raw_tail / raw_tail.sum()) * target_sum
```

This produces a right-skewed tail with the correct *shape*, individually rescaled so its share of total volume hits the intended number.

One caveat worth stating rather than glossing: the rescale itself is exact, but the integer rounding and the floor clip applied immediately afterwards can both move the sum away from the target. At the parameters in this repo the drift happens to be 0.0 percentage points — but that is something the script now *prints and checks on every run*, rather than something the reader is asked to take on trust. Change the segment share or the tail shape and the drift may stop being zero, which is precisely why it is measured rather than assumed.

## Where this went wrong the first time, and what I learned from it

The generation approach above isn't how the dataset started — it's what I converged on after catching real problems in an earlier version.

**Problem 1: captions didn't match the actual generated numbers.** An early draft had two of four illustrative charts showing figures I'd hand-picked as placeholders before any data existed, alongside two charts whose captions correctly recalculated from the real generated array. The two didn't agree — off by roughly ten percentage points. The fix wasn't to nudge the placeholder text closer to the real number; it was to make every caption a computed value, sourced from the same array the chart itself was built from, so a mismatch becomes structurally impossible rather than something to remember to check.

**Problem 2: "stable" data that was suspiciously stable.** A stress-test panel was meant to show a monitored metric holding steady across a monitoring window. The first version showed it landing on the exact same value, to the decimal, in every single period — mathematically consistent, but not something real data ever actually looks like. Zero variance across six independent months reads as fabricated to anyone who's looked at real operational data before. The fix was deliberately reintroducing controlled random variation — enough that the metric moved period to period the way genuine data does, while still never crossing the threshold the scenario was built to demonstrate.

**Problem 3: a stale number surviving two edit passes.** After fixing problem 1, a later editing pass focused on visual formatting (axis labels, spacing) — not content. A full re-verification afterward, checking *every* number again rather than just the ones touched in that pass, turned up a figure that had never actually been checked against the real data at all, from before problem 1 was even discovered. It had survived because every prior check had only re-verified numbers that had just changed.

## The actual lesson

None of these are exotic statistics problems. They're a discipline problem: generated content needs the same verification a real analysis would get, not less — arguably more, since there's no original source of truth to fall back on and catch a drifted number. The script in this repo (`generate_data.py`) prints every computed statistic to the console specifically so it can be diffed against whatever prose references it, every time it's regenerated — not just the first time.

## Running it

```bash
pip install numpy matplotlib
python generate_data.py
```

Outputs four chart PNGs plus a console printout of every underlying statistic — median, percentile table, coverage percentages, and volume concentration — meant to be checked against any accompanying written content before publishing either.
