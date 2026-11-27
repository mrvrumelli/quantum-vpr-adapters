# Stage 3.2 · Residual and circuit contribution

**The quantum branch changes the descriptors, and the learned angles matter within some trained checkpoints. Training the PQC provided no reliable retrieval benefit over training the same projections with a frozen circuit, or over the matched classical residual controls.** At the strongest observed quantum scale, λ=0.50, mean validation MRR was 0.58311 with learned angles and 0.58323 with frozen angles. This experiment gives no empirical reason to increase circuit expressivity.

## Protocol

This study used the frozen 512-dimensional CosPlace caches, 80 training query–gallery pairs, and the same 50 validation queries and 50 gallery images. Only the training triplets contributed gradients. The existing cosine ranking, Recall@1/5/10, and full-gallery reciprocal rank were used. The held-out test split stayed untouched.

All trained arms used the Stage 3.1 `512 → 64 → 4 → 512` residual model, LeakyReLU with slope 0.01 at the bottleneck, and a bias-free output projection initialized at 0.05 times the seeded linear initialization. Each seed used equal initial projection weights across all arms, verified bitwise. A, B, and C also shared the same starting angle values. Removing CNOTs changes the circuit's initial function. The surrogate and the identity function also have different initial outputs from the PQC.

| Arm | Bottleneck transformation | Classical / quantum trainable |
| --- | --- | ---: |
| A | Two-layer trainable PQC with CNOT chain | 35,140 / 16 |
| B | Same PQC with seeded angles frozen and both projections trained | 35,140 / 0 |
| C | Two-layer trainable PQC with no CNOTs | 35,140 / 16 |
| D | Stage 3.1 surrogate `tanh(W tanh(z))`, with a bias-free 4 × 4 matrix | 35,156 / 0 |
| E | Identity mixer, with both surrounding projections trained | 35,140 / 0 |

Adam used the same learning rate of 0.001, triplet margin of 0.2, batch size of 16, exclusion window of five frames, and 50 epochs with patience 50. Matching seeds had the same triplets and order. The earliest lexicographic best validation Recall@1/5/10 selected the checkpoint. Seeds were `42, 123, 2026, 7, 31415`. The λ=0 result is untrained CosPlace. The first pass crossed five seeds with the quantum scale sweep and the controls at λ=0.10. The final grid crosses **all** four nonzero scales with **all** five controls and five seeds, so each scale comparison has its own matched controls. A learnable scale was omitted to keep the fixed-scale experiment interpretable.

The **angle reset** changes only the angles at inference, replacing the selected checkpoint's learned parameters with the initial values for that seed and keeping its trained projections fixed. The **frozen PQC** instead trains the projections with fixed angles from the start. Readout differences for the reset are measured using the same projected inputs. Per-epoch gradient norms are means of minibatch L2 norms. The input-gradient column measures the loss gradient at the four mixer inputs, accumulated over the query, positive, and negative forwards in a batch. The projection gradient column measures the weights and biases of the two input projection layers.

The paired analysis computes the per-query differences between models trained with the same seed. For each of 20,000 seeded bootstrap draws, it samples five paired seeds and ten contiguous five-frame validation blocks with replacement. All sampled seeds share the same sampled blocks in a draw. The ten-frame sensitivity analysis uses five larger blocks. The 95% percentile intervals are descriptive and unadjusted across comparisons. They are conditional on this route, gallery, and checkpoints selected on validation. Adjacent frames may remain correlated across blocks. Five seeds on the same 50 queries do not provide 250 independent scenes. One query changes a single seed's Recall@K by two percentage points.

## Retrieval and scale

All **100** final runs completed 50 epochs. The three original Stage 3.1 Quantum Residual checkpoints matched the selected epochs, every checkpoint tensor, and all 390 train and validation first-positive ranks for those seeds. The additional runs used the same frozen caches. Retrieval means and sample standard deviations use five seeds. λ=0 is a single deterministic baseline.

| λ | Quantum R@1 | Quantum R@5 | Quantum R@10 | Descriptor effective rank | Descriptor variance | Mean scaled correction norm | Mean cosine to CosPlace |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 30.0% | 92.0% | 94.0% | 34.84 | 0.81177 | 0 | 1.00000 |
| 0.05 | 33.2 ± 1.1% | 91.6 ± 0.9% | 94.8 ± 1.1% | 34.79 | 0.81769 | 0.06913 | 0.99672 |
| 0.10 | 33.6 ± 0.9% | 92.0 ± 0.0% | 94.4 ± 0.9% | 34.72 | 0.81984 | 0.10203 | 0.99288 |
| 0.20 | 34.8 ± 1.1% | 90.8 ± 1.8% | 94.4 ± 0.9% | 34.49 | 0.81969 | 0.15558 | 0.98362 |
| 0.50 | 35.2 ± 1.1% | 91.2 ± 1.1% | 94.0 ± 0.0% | 34.63 | 0.79937 | 0.21033 | 0.97616 |

| λ | A · Trainable PQC | B · Frozen PQC | C · No CNOTs | D · Matched classical | E · Identity mixer | Angles reset |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.57350 ± 0.00482 | 0.57226 ± 0.00412 | 0.57772 ± 0.00345 | 0.56955 ± 0.00957 | 0.57659 ± 0.00252 | 0.57055 ± 0.00848 |
| 0.10 | 0.57503 ± 0.00537 | 0.57182 ± 0.00737 | 0.57633 ± 0.00667 | 0.57153 ± 0.00999 | 0.57718 ± 0.00228 | 0.57500 ± 0.00535 |
| 0.20 | 0.57667 ± 0.00793 | 0.57524 ± 0.00419 | 0.57002 ± 0.00392 | 0.57668 ± 0.00288 | 0.57767 ± 0.00273 | 0.56114 ± 0.00619 |
| 0.50 | 0.58311 ± 0.00669 | 0.58323 ± 0.00672 | 0.57757 ± 0.00953 | 0.58158 ± 0.00792 | 0.58119 ± 0.01529 | 0.56621 ± 0.00850 |

CosPlace MRR was **0.55626**. A larger residual increased the trainable PQC's mean R@1 and MRR in this sweep, with mixed changes at R@5 and R@10. The λ=0.50 arm gained `+0.02685` MRR over CosPlace with a five-frame interval of `[+0.00206, +0.05389]`. Its paired gain over λ=0.10 was `+0.00808` with an interval of `[−0.01010, +0.02901]`. The frozen circuit nearly duplicated the strongest observed quantum mean. Both classical controls also improved their average MRR at the larger scale over their λ=0.10 runs.

### Paired validation effects

| λ | A minus | Δ MRR | Across-seed SD | 95% interval, five-frame blocks |
| ---: | --- | ---: | ---: | ---: |
| 0.10 | B · Frozen PQC | +0.00322 | 0.00546 | [−0.00164, +0.01343] |
| 0.10 | C · No CNOTs | −0.00130 | 0.00931 | [−0.01287, +0.00881] |
| 0.10 | D · Matched classical | +0.00350 | 0.01281 | [−0.01069, +0.01930] |
| 0.10 | E · Identity mixer | −0.00215 | 0.00611 | [−0.01518, +0.00680] |
| 0.10 | Angles reset | +0.00003 | 0.00006 | [0.00000, +0.00013] |
| 0.50 | B · Frozen PQC | −0.00012 | 0.00485 | [−0.01598, +0.01322] |
| 0.50 | C · No CNOTs | +0.00554 | 0.01097 | [−0.01821, +0.02227] |
| 0.50 | D · Matched classical | +0.00153 | 0.00913 | [−0.01738, +0.01989] |
| 0.50 | E · Identity mixer | +0.00193 | 0.01639 | [−0.02544, +0.02628] |
| 0.50 | Angles reset | +0.01690 | 0.00895 | [−0.00283, +0.04126] |

At each scale, the MRR intervals for A versus all four independently trained controls included zero under both block sizes. The paired CSV includes every scale and every recall cutoff. [Paired-effect figure](results/mrr-comparisons.svg) · [Numerical intervals](results/paired_bootstrap.csv)

## Residual contribution at λ=0.10

| Split | Mean original norm | Mean unscaled correction | Mean scaled correction | Mean original–adapted cosine | First-positive ranks improved / worsened / same |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1.00000 | 1.23511 | 0.12351 | 0.98937 | 45 / 0 / 355 |
| Validation | 1.00000 | 1.02032 | 0.10203 | 0.99288 | 19 / 13 / 218 |

The descriptor statistics pool the query and gallery within each split and average across seeds. Rank counts compare with unadapted CosPlace and count repeated query–seed observations. On validation the mean of the within-seed 5th and 95th percentiles of the scaled correction norm was 0.04728 and 0.15805. The original descriptor dominates the output direction, while the small correction changes some ranks. The mean first-positive rank stayed the same even as MRR rose by 0.01877, which reflects the greater weight MRR places on changes near the top.

| Validation pair type | Mean cosine change | Mean within-seed 5th / 95th percentile | Increased / decreased |
| --- | ---: | ---: | ---: |
| Labeled positive | −0.00550 | −0.01976 / +0.01096 | 62 / 188 |
| Eligible negative | −0.00994 | −0.02869 / +0.00861 | 1,910 / 7,990 |

Positives and negatives both moved down on average, with a larger decline for negatives. Eligible negatives exclude the same five-frame window used in training. The positive-minus-negative mean separation increased by 0.00444 on validation. On training it increased by 0.02399, with mean shifts of −0.01359 for positives and −0.03757 for negatives. This is a change in score separation on this route. It does not establish that such separation requires the circuit. [Per-run distributions](results/runs.csv) · [Query-level similarity and rank effects](results/query_effects.csv)

## Quantum learning diagnostics

The norms below average the minibatches in each selected epoch, then average across the five seeds. The CSV retains all 50 epochs for each run, along with per-wire angle updates and readout covariance on both splits.

| λ | Layer 1 angle gradient | Layer 2 angle gradient | Mixer input gradient | Val readout total variance | Val readout effective rank |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.000344 | 0.000575 | 0.002068 | 0.21038 | 1.218 |
| 0.10 | 0.001627 | 0.001836 | 0.005887 | 0.12075 | 1.217 |
| 0.20 | 0.001035 | 0.000963 | 0.005543 | 0.14555 | 1.212 |
| 0.50 | 0.000754 | 0.000412 | 0.005667 | 0.04759 | 1.392 |

At λ=0.10, the selected angles moved by a median absolute 0.0140 rad in layer 1 and 0.0173 rad in layer 2, with maxima 0.0736 and 0.0909 rad. Validation readout variance, computed as the trace of centered population covariance, rose from 0.00252 to 0.12075. Readout effective rank stayed close to one (1.34 initially and 1.22 after selection). The four Pauli-Z coordinates had mean validation variances of 0.05250, 0.01816, 0.02643, and 0.02366. The frozen arm reached a similar variance of 0.12122 and effective rank of 1.21, so shifts in the input projections can explain rising readout variance even with fixed angles. Both layers had nonzero gradients. These measurements do not establish a barren plateau.

At λ=0.10, resetting the angles on the same projected inputs changed the Pauli-Z readout by 0.02768 RMS. Removing its mean shift left 0.00922 RMS of sample-dependent change, compared with 0.16506 RMS centered variation in the learned readout. The reset changed just **2 of 250** paired first-positive ranks and left all Recall@K values the same. At λ=0.20 and 0.50 it changed **35 and 59 ranks**, respectively. The learned angles added `+0.01553` MRR over reset at λ=0.20 in this fixed-projection comparison, with intervals `[+0.00081, +0.03644]` and `[+0.00085, +0.03396]` for the two block sizes. At λ=0.50 the reset effect was `+0.01690`, and inclusion of zero depended on block length. This shows that checkpoints at larger scales can rely on their learned angles. The independently optimized frozen models can attain comparable retrieval, consistent with their surrounding projections adapting to the fixed transformation.

## Research assessment

1. **Trainable versus frozen.** Training the angles brought no reliable end-to-end gain. The reset does establish a role for the learned angles within some fixed checkpoints, especially at λ=0.20.
2. **Entanglement.** The CNOT arm had higher MRR at λ=0.20 and 0.50 and lower MRR at λ=0.05 and 0.10. The paired intervals leave the contribution of entanglement unresolved.
3. **Residual contribution.** The branch preserves the overall descriptor direction at λ=0.10 and produces a measurable redistribution of query ranks and lower negative cosine scores. Larger scales produce larger changes. The same-scale controls show these effects are achievable with fixed or classical transformations.
4. **Classical comparison.** The PQC has no consistent advantage over the parameter-matched surrogate or the identity mixer with learned projections. The parameter count match holds for the surrogate. It does not equate the compute costs of simulating a circuit and applying a dense layer.
5. **Expressivity.** The higher-priority next research step is replication on independent routes or held-out data after freezing the protocol. This experiment alone does not warrant a larger circuit. The held-out test split was kept untouched during the scale and architecture comparisons.

## Reproduction and exports

Run from the repository root with the existing Gardens Point caches. The optional archive flag checks the original Stage 3.1 selected epochs and train/validation first-positive ranks. Each run writes checkpoints and histories to a new local artifact directory.

```bash
python -m pip install -e '.[diagnostics]'
OMP_NUM_THREADS=1 COSPLACE_CHECKPOINT=checkpoints/resnet18_512_cosplace.pth \
  python -m unittest discover -s tests -v

OMP_NUM_THREADS=1 python experiments/quantum-stage-3-2/run_study.py \
  --dataset-root data/GardensPointWalking --cache-dir artifacts/gardens-point \
  --output-dir artifacts/quantum-stage-3-2-replay \
  --results-dir artifacts/quantum-stage-3-2-replay-results \
  --stage-31-dir artifacts/quantum-stage-3-1

python experiments/quantum-stage-3-2/plot_comparisons.py \
  artifacts/quantum-stage-3-2-replay-results/paired_bootstrap.csv \
  artifacts/quantum-stage-3-2-replay-results/mrr-comparisons.svg
```

- [Exact configuration](results/protocol.json), [per-run train and validation measurements](results/runs.csv), and [mean ± sample SD across seeds](results/summary.csv).
- [Paired confidence intervals](results/paired_bootstrap.csv), [per-seed differences](results/paired_seeds.csv), [per-query paired results](results/paired_queries.csv), and [each model's query effects versus CosPlace](results/query_effects.csv).
- [Per-image residual norms at λ=0.10](results/residual_norms.csv), [readout covariance](results/feature_covariance.csv), [angle update distributions](results/angle_updates.csv), [fixed-projection angle reset](results/angle_reset_features.csv), and [epoch-level training and gradient traces](results/training_trace.csv).
- [Stage 3.1 checkpoint checks](results/stage_3_1_checkpoints.csv) and [paired MRR figure](results/mrr-comparisons.svg).
