# Phase Vb — source-sensitivity control

This lightweight control was added after Phase V to determine whether the significant artist-centroid style effect in the full ArtBench-10 pilot is disproportionately driven by the two styles whose image sources differ from the main WikiArt-derived classes: `surrealism` and `ukiyo_e`.

No image extraction or classifier fitting is required. The analysis reuses `artbench_pilot_features.csv` and exactly the same artist-centroid permutation test used in Phase V.

## Prespecified variants

The multivariate style test is repeated at `sigma_ref = {1, 2, 4, 8}` for:

1. all 10 ArtBench styles;
2. all styles except `ukiyo_e`;
3. all styles except `surrealism`;
4. the WikiArt-8 sensitivity subset excluding both.

The primary scale of interest is `sigma_ref = 2`, because it was the strongest style-organizing scale in Phase V.

## Results at sigma_ref = 2

| Variant | Single-style artists | Styles | pseudo-F | eta^2 style | permutation p |
|---|---:|---:|---:|---:|---:|
| All 10 | 179 | 10 | 2.2445 | 0.1068 | 0.0140 |
| Drop Ukiyo-e | 164 | 9 | 1.7708 | 0.0837 | 0.0506 |
| Drop Surrealism | 152 | 9 | 1.3329 | 0.0694 | 0.1802 |
| Drop both (WikiArt-8) | 137 | 8 | 0.9669 | 0.0499 | 0.4420 |

Relative to the full ten-style effect size, eta^2 decreases by approximately 21.6% after removing Ukiyo-e, 35.0% after removing Surrealism, and 53.3% after removing both source-specific classes.

## Leave-one-style-out diagnostic

As an exploratory diagnostic, the sigma=2 test was repeated after removing each of the ten styles in turn. Removing any of the eight non-source-specific styles leaves the effect significant (`p <= 0.0134`) with `eta^2 >= 0.0951`. The largest attenuation occurs when removing `surrealism`, followed by `ukiyo_e`.

This diagnostic should not be interpreted as ten independent confirmatory hypothesis tests. Its role is to identify whether the full-corpus effect is unusually sensitive to particular classes.

## Interpretation

The significant sigma=2 centroid separation in the full ArtBench-10 pilot is not robust to removal of the two source-specific styles. The effect is especially sensitive to `surrealism`, although both source-specific classes contribute to the attenuation. Therefore the manuscript should not use the full ten-style centroid test as standalone evidence that artistic movements form statistically distinct geometric clusters.

The more defensible conclusions remain:

- level-set geometry carries predictive style information under artist-disjoint evaluation;
- fine/intermediate scales, particularly sigma around 2, contain most of that information;
- the amount of style-level geometric organization is modest relative to artist- and painting-level variability;
- conclusions about between-style centroid separation are sensitive to dataset/source composition.

Reproduce with:

```bash
python scripts/run_phase5b_source_sensitivity.py \
  --features /path/to/artbench_pilot_features.csv \
  --output-dir results/phase5b_source_sensitivity \
  --n-permutations 4999 \
  --seed 42
```
