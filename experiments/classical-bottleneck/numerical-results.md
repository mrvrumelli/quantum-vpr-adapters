# Exact bottleneck sweep results and optimization audit

This export covers the 24 runs with intermediate width 64, bottleneck outputs 4, 8, 16, and 32, and seeds 42, 123, and 2026. Training used 80 queries and 80 gallery images. Validation used 50 of each. Each run used the existing 50-epoch protocol. I recomputed train and validation recall from every saved selected checkpoint and matched the results to the archived diagnostics. All 24 selected epochs also matched the earliest lexicographically best validation Recall@1/5/10 tuple in their recorded histories. The test split stayed held out.

| File | Rows | Contents |
| --- | ---: | --- |
| [Ablation runs](ablation_runs.csv) | 24 | Architecture, widths, seed, selected epoch, parameter count, and train and validation Recall@1/5/10 as fractions |
| [Descriptor diagnostics](ablation_diagnostics.csv) | 48 | One train and one validation row per run with geometry, positive and negative cosine statistics, and their mean gap |
| [Residual diagnostics](residual_diagnostics.csv) | 24 | Per-seed and per-split alpha, alignment, correction strength, and first-positive rank changes |
| [Gradient audit](gradient_audit.csv) | 48 | Initialization and selected-checkpoint probes with active triplets, layer gradient norms, ReLU inactivity, and normalization checks |

## Validation Recall@K across seeds

Values are mean recall in percent ± sample standard deviation in percentage points across the three seeds. The per-run CSV keeps the original recall fractions. [Full-precision seed summary](seed_summary.csv)

| Architecture | Bottleneck | Parameters | R@1 (%) | R@5 (%) | R@10 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct | 4 | 35,652 | 11.3333 ± 1.1547 | 28.6667 ± 2.3094 | 45.3333 ± 2.3094 |
| Direct | 8 | 37,960 | 10.6667 ± 4.1633 | 26.6667 ± 3.0551 | 44.0000 ± 6.9282 |
| Direct | 16 | 42,576 | 12.0000 ± 5.2915 | 26.6667 ± 6.4291 | 42.6667 ± 9.8658 |
| Direct | 32 | 51,808 | 16.0000 ± 2.0000 | 36.6667 ± 7.0238 | 55.3333 ± 6.4291 |
| Residual | 4 | 35,652 | 33.3333 ± 5.0332 | 88.0000 ± 2.0000 | 94.6667 ± 1.1547 |
| Residual | 8 | 37,960 | 35.3333 ± 2.3094 | 90.0000 ± 2.0000 | 94.6667 ± 2.3094 |
| Residual | 16 | 42,576 | 34.6667 ± 1.1547 | 88.0000 ± 3.4641 | 95.3333 ± 1.1547 |
| Residual | 32 | 51,808 | 35.3333 ± 1.1547 | 89.3333 ± 1.1547 | 95.3333 ± 1.1547 |

## Descriptor diagnostics

Effective rank and total variance are means of the three per-seed values, calculated on pooled query and gallery descriptors within each run. Total variance is the trace of the population covariance of unit vectors. Effective rank is the exponential entropy of the normalized squared singular values of centered descriptors. The positive and negative columns pool pair scores across the three seed runs using their means, population variances, and counts. The mean gap is positive mean minus negative mean. Each seed uses the same image pairs, so these pooled SDs are descriptive. A negative is a gallery frame outside the existing five-frame exclusion window for the query's positives. [Per-run diagnostics](ablation_diagnostics.csv) · [Original CosPlace rows](diagnostics.csv)

| Architecture | Bottleneck | Split | Effective rank | Variance | Positive cosine mean ± SD | Negative cosine mean ± SD | Mean gap |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CosPlace | — | train | 40.7569 | 0.811373 | 0.571135 ± 0.094709 | 0.122119 ± 0.116556 | 0.449016 |
| CosPlace | — | val | 34.8407 | 0.811772 | 0.318179 ± 0.120851 | 0.075935 ± 0.070189 | 0.242243 |
| Direct | 4 | train | 2.5917 | 0.577579 | 0.993850 ± 0.009664 | 0.346804 ± 0.380238 | 0.647047 |
| Direct | 4 | val | 2.2323 | 0.159541 | 0.924659 ± 0.083971 | 0.824843 ± 0.153827 | 0.099816 |
| Direct | 8 | train | 2.7795 | 0.493360 | 0.990802 ± 0.011638 | 0.443808 ± 0.463809 | 0.546994 |
| Direct | 8 | val | 2.4574 | 0.141617 | 0.926734 ± 0.098532 | 0.841179 ± 0.194209 | 0.085555 |
| Direct | 16 | train | 3.0484 | 0.441826 | 0.990238 ± 0.015059 | 0.500338 ± 0.402573 | 0.489900 |
| Direct | 16 | val | 2.9938 | 0.100681 | 0.942990 ± 0.073752 | 0.890345 ± 0.135477 | 0.052644 |
| Direct | 32 | train | 3.2040 | 0.303111 | 0.990333 ± 0.014102 | 0.656930 ± 0.396747 | 0.333403 |
| Direct | 32 | val | 5.7447 | 0.078029 | 0.963125 ± 0.058281 | 0.913786 ± 0.133802 | 0.049340 |
| Residual | 4 | train | 38.9897 | 0.623384 | 0.685340 ± 0.063302 | 0.323735 ± 0.094146 | 0.361605 |
| Residual | 4 | val | 34.6222 | 0.598752 | 0.498985 ± 0.091595 | 0.318223 ± 0.058147 | 0.180761 |
| Residual | 8 | train | 38.4200 | 0.718872 | 0.641371 ± 0.071297 | 0.220064 ± 0.103288 | 0.421307 |
| Residual | 8 | val | 34.4035 | 0.677812 | 0.436706 ± 0.102526 | 0.229555 ± 0.068875 | 0.207151 |
| Residual | 16 | train | 38.2863 | 0.796484 | 0.608359 ± 0.072826 | 0.134802 ± 0.107246 | 0.473557 |
| Residual | 16 | val | 34.3024 | 0.757985 | 0.372390 ± 0.114831 | 0.137801 ± 0.072046 | 0.234589 |
| Residual | 32 | train | 38.2618 | 0.865912 | 0.578847 ± 0.079400 | 0.060377 ± 0.119691 | 0.518470 |
| Residual | 32 | val | 34.2611 | 0.787709 | 0.349602 ± 0.123113 | 0.104512 ± 0.078520 | 0.245090 |

### Does Direct collapse at every tested width and seed?

**Partial collapse persists in all 12 Direct runs on both splits, with different severity.** On training data, every Direct run has lower effective rank and variance than CosPlace. The effective ranks are 1.55–4.46, and mean distinct-image cosines range from 0.277 to 0.969. Several runs therefore retain substantial training variation in a few directions. On validation, all Direct runs have effective ranks of 1.40–9.35, total variance of 0.0061–0.2153, and mean distinct-image cosine of 0.783–0.994. CosPlace has 34.84, 0.8118, and 0.180, respectively. All 12 Direct validation mean gaps are below the CosPlace gap of 0.2422, ranging from 0.0027 to 0.1640.

Width 32 has the strongest average validation Recall@K among Direct layouts. One width-32 seed illustrates the difference between rank and spread. Seed 2026 has the highest Direct validation effective rank, 9.35, and the lowest validation variance, 0.0061. That checkpoint distributes its remaining variance across more directions inside a very tight angular cluster. Wider bottlenecks can relieve the rank constraint, and the results show that widening alone leaves substantial angular concentration.

## Residual alignment, scale, and ranks

The alignment column averages cos(z, z′) over queries and gallery images, then over the three seeds. Relative correction averages ‖αΔz‖/‖z‖ in the same way. Alpha is the fixed value loaded from each checkpoint. Counts sum how often the first labeled positive moves up, down, or stays at the same rank in the full gallery relative to frozen CosPlace. Each row covers 240 training or 150 validation query evaluations across seeds. [Per-seed residual diagnostics](residual_diagnostics.csv)

| Bottleneck | Split | α | Mean cos(z, z′) | Mean relative correction | Improved / worsened / unchanged |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4 | train | 0.1 | 0.829803 | 0.654358 | 31 / 6 / 203 |
| 4 | val | 0.1 | 0.843326 | 0.630009 | 18 / 24 / 108 |
| 8 | train | 0.1 | 0.884713 | 0.511272 | 37 / 6 / 197 |
| 8 | val | 0.1 | 0.901668 | 0.477275 | 24 / 20 / 106 |
| 16 | train | 0.1 | 0.920322 | 0.408781 | 42 / 3 / 195 |
| 16 | val | 0.1 | 0.945796 | 0.339892 | 27 / 21 / 102 |
| 32 | train | 0.1 | 0.927155 | 0.373712 | 48 / 5 / 187 |
| 32 | val | 0.1 | 0.952970 | 0.308402 | 24 / 23 / 103 |

## Optimization audit

For each run I reconstructed its seeded initialization under PyTorch 2.11.0 and loaded its selected weights. The probe used the run's first seeded sample of 80 train triplets, split into the original five 16-item batches. The same batches were applied to both snapshots, with weights fixed throughout. Each layer's gradient statistic is the RMS across batches of the joint L2 norm of its weight and bias gradients. The probe measures these two snapshots. The full optimizer trajectory would require gradient logs from the original runs. [Probe CSV](gradient_audit.csv)

| Architecture | Snapshot | Active triplets out of 80, range | Median down gradient | Median bottleneck gradient | Median up gradient |
| --- | --- | ---: | ---: | ---: | ---: |
| Direct | Initialization | 80–80 | 0.009803 | 0.003902 | 0.000223 |
| Direct | Selected | 1–80 | 0.055858 | 0.045579 | 0.008542 |
| Residual | Initialization | 3–16 | 0.003028 | 0.002818 | 0.005208 |
| Residual | Selected | 0–16 | 0.003209 | 0.003271 | 0.001812 |

**Observed in the code and probes**

- **Objective.** The hinge is `max(0, 0.2 + cosine(q, n) − cosine(q, p))`. The sign penalizes negatives above the required margin. Direct starts with all 80 probe triplets active and a mean hinge of 0.1981–0.1999. At the selected snapshots the median active count is 10. Thus the sampled objective can decrease substantially even when validation descriptors remain concentrated. [Training objective and sampler](../../src/vpr_baseline/training.py)
- **Normalization.** The forward pass divides by each row's L2 norm while keeping gradients connected. Cached input norm errors were at most 1.48×10⁻⁷, and the largest output norm error across the audited train and validation snapshots was 2.12×10⁻⁷. The smallest pre-normalization norm measured on training rows was 0.875. [Adapter forward pass](../../src/vpr_baseline/adapters.py) · [L2 normalization](../../src/vpr_baseline/retrieval.py)
- **Initialization.** The matched adapters start with identical MLP weights for each width and seed. PyTorch's default linear initialization uses weights and biases bounded by `±1/√fan_in`. In every reconstructed initialization, the up-projection bias norm exceeded the RMS norm of the input-dependent linear term. Their medians were 4.019 and 0.635, respectively. Direct's initial validation distinct-image cosine was already 0.9968–0.9998. The skip connection gives Residual original input directions even with the same initial MLP.
- **Gradients.** Every Direct run had finite, nonzero aggregate gradients in all three layers at both snapshots. At initialization every Direct run had 1–12 ReLU units inactive on all training images. Ten of 12 selected checkpoints still had at least one such unit. Each inactive unit received exactly zero gradient in every probe batch. Four selected Residual runs had zero gradients across the probe because all sampled triplets satisfied the margin.

**Possible causes and limits.** The Direct output lies in the span of the up-projection's bottleneck columns and its bias, with dimension at most `bottleneck + 1`. Normalization preserves that span. The bound explains why Direct cannot carry the full CosPlace geometry by construction. A dominant common bias can also make normalized outputs point in nearly the same direction. Inactive ReLUs reduce the number of available directions further, but two selected Direct runs retain partial collapse even with every unit active somewhere on training data. The hinge optimizes sampled comparisons and has no explicit term for keeping descriptors spread out. As a sanity check, three identical normalized directions had a positive hinge of 0.2 within float precision and exactly zero input gradient. This establishes that a collapsed direction can be stationary under the objective. The existing runs and fixed probes cannot assign separate causal effects to initialization, ReLU inactivity, or the objective. The experiment settings stayed fixed throughout this audit.
