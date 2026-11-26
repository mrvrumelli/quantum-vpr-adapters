# Direct up-projection bias ablation

Removing the final projection bias reduced validation angular concentration in 10 of the 11 completed pairs. All eight completed pairs from the fixed-intermediate sweep had higher descriptor variance and effective rank. Five bias-free runs produced zero descriptors, including every seed for the `64 → 4` layout. Recall gains varied by width.

## Matched comparison

The 16 pairs used the archived Direct run configurations. These came from the 12 runs with intermediate width 64 and bottleneck dimensions 4, 8, 16, and 32, the three `8 → 4` reference repeats, and the original Phase 2 `8 → 4` seed-0 run. Each pair shared the frozen train and validation caches, seeded initial weights and biases in every shared parameter, training triplet sequence, optimizer, schedule, and checkpoint selection criterion. The sweep used the archived 50-epoch budget and patience. The original seed-0 pair used its archived early-stopping settings. All 16 replayed bias-on histories exactly matched the archived histories. The test split stayed held out.

The sole model change was omitting the 512 bias parameters of the up-projection. The final L2 normalization still requires nonzero vectors. Initialization metrics use identical rows with nonzero descriptors in both runs. Selected-checkpoint metrics use the full split when both runs completed. [Per-seed train and validation CSV](direct_bias_runs.csv)

## Validation results

Each cell shows bias-on → bias-off. Means for both arms use only the seed pairs that completed. Recall values are percentages over the same 50 validation queries and 50 gallery images per run.

| Intermediate → bottleneck | Completed pairs | Distinct-image cosine | Total variance | Effective rank | Recall@1 / @5 / @10 (%) |
| --- | ---: | ---: | ---: | ---: | --- |
| 8 → 4, original seed 0 | 1/1 | 0.9956 → 0.8656 | 0.0043 → 0.1331 | 1.58 → 1.94 | 8 / 12 / 18 → 4 / 12 / 22 |
| 8 → 4, reference repeats | 2/3 | 0.9152 → 0.8654 | 0.0840 → 0.1332 | 2.12 → 1.85 | 13 / 31 / 50 → 11 / 23 / 34 |
| 64 → 4 | 0/3 | — | — | — | — |
| 64 → 8 | 2/3 | 0.7896 → 0.6787 | 0.2083 → 0.3181 | 2.99 → 3.35 | 13 / 28 / 46 → 9 / 26 / 42 |
| 64 → 16 | 3/3 | 0.8983 → 0.6060 | 0.1007 → 0.3901 | 2.99 → 5.20 | 12.00 / 26.67 / 42.67 → 14.67 / 33.33 / 49.33 |
| 64 → 32 | 3/3 | 0.9212 → 0.7203 | 0.0780 → 0.2769 | 5.74 → 8.96 | 16.00 / 36.67 / 55.33 → 14.00 / 38.67 / 52.67 |

Across the 11 completed pairs, mean validation cosine fell from 0.8967 to 0.7211 and mean effective rank rose from 3.46 to 4.98. Mean Recall@1 moved from 13.09% to 11.82%. The `64 → 16` layout was the only layout that completed every seed and increased mean recall at all three cutoffs. Its bias-free means remained below the original CosPlace recalls of 30%, 92%, and 94%.

At initialization, mean pairwise cosine fell in all 16 pairs on both splits among rows valid in both arms. Effective rank fell in 15 of 16 pairs on each split. Removing a common direction increases angular spread, and the spread can still be concentrated in a few directions. The bias-free up-projection also has a linear rank bound of the bottleneck width. The selected checkpoints show that angular spread and effective rank can improve together after training.

## Zero descriptors

Replaying each failed run reproduced the first zero-norm batch. In every zero row, all bottleneck ReLU outputs were zero. The bias-free up-projection maps that row to exactly zero, which the existing cosine pipeline cannot normalize. The recorded failed runs have blank selected-checkpoint metrics.

| Intermediate → bottleneck | Seed | Epoch | Stage | Zero rows in the failing forward pass |
| --- | ---: | ---: | --- | ---: |
| 8 → 4 | 42 | 1 | Train | 9/16 |
| 64 → 4 | 42 | 3 | Train | 1/32 |
| 64 → 4 | 123 | 43 | Train | 1/16 |
| 64 → 4 | 2026 | 1 | Train | 1/32 |
| 64 → 8 | 2026 | 33 | Validation | 1/50 |

The intervention implicates the up-projection bias as a contributor to angular concentration in the measured runs. The zero-vector failures and mixed recall results limit bias removal as a standalone change, especially for the four-dimensional bottleneck.

## Reproduce

The checked-in CSV was produced on CPU with one thread. Use the existing Phase 2 and bottleneck-ablation histories to recover the exact settings for each pair.

```bash
OMP_NUM_THREADS=1 python experiments/classical-bottleneck/run_direct_bias_ablation.py \
  --dataset-root data/GardensPointWalking \
  --cache-dir artifacts/gardens-point \
  --control-dir artifacts/classical-bottleneck/runs \
  --phase2-dir artifacts/gardens-point-adapters \
  --output-dir artifacts/direct-up-bias
```

The `train-adapter --adapter direct --no-up-bias` CLI option also supports an individual run. New checkpoints record the bias choice. Historical checkpoints load with the original bias enabled.
