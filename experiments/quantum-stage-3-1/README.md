# Stage 3.1 · Four-qubit adapter benchmark

Both quantum adapters trained successfully under the Stage 2.3 protocol. Quantum Residual reached mean validation MRR **0.5723**, close to the matched classical residual surrogate (**0.5709**) and the archived Residual MLP (**0.5690**). At seed 42 a frozen randomly initialized circuit reproduced all 50 first-positive ranks of the trainable Quantum Residual. The residual connection is the most consistent feature of the stronger models in this comparison. These results give no evidence of a quantum advantage.

## Circuit and adapter interface

Both quantum variants use the frozen 512-dimensional CosPlace descriptor `x`. The trainable outer projection is `512 → 64 → 4`, with LeakyReLU of slope 0.01 after the second linear layer as in classical control C. The common angle rule is `θᵢ = π tanh(zᵢ)`. This bounds the angle to one rotation interval, leaves a nonzero derivative near zero, and avoids periodic wrapping of arbitrarily large inputs. The validation diagnostics measure saturation as the fraction of angles with absolute value at least `0.95π`.

The simulator starts from `|0000⟩` and uses bit `i` of the basis index for wire `i`.

1. Apply `RY(θᵢ)` on wires 0, 1, 2, 3.
2. In each layer, on wires 0, 1, 2, 3 in order, apply trainable `RZ(φₗᵢ)` followed by trainable `RY(βₗᵢ)`.
3. End each layer with `CNOT(0,1)`, `CNOT(1,2)`, `CNOT(2,3)` in that order. The separable ablation omits these gates.
4. Return the four exact Pauli-Z expectations from the output probabilities. Small floating-point overshoots at ±1 are clamped to that interval.

The default depth is two, with 16 rotation angles. Initialize `φ` uniformly in `[-0.1, 0.1]` and `β` in `π/4 + [-0.1, 0.1]`. Angle indexing is `[layer, wire, (RZ, RY)]`. The statevector is a batched PyTorch tensor with 16 complex amplitudes per image. PyTorch reverse-mode autograd differentiates the exact statevector and the encoders within the existing CPU training step. This keeps the dependency set and tensor interface small. [PyTorch 2.11 complex autograd](https://docs.pytorch.org/docs/2.11/notes/autograd.html#autograd-for-complex-numbers) · [RY definition](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RYGate) · [RZ definition](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RZGate)

| Stage | Shape for a batch of size B |
| --- | --- |
| Frozen descriptors | `[B, 512]` |
| First projection | `[B, 64]` |
| Bottleneck and angle encoding | `[B, 4]` |
| Complex statevector | `[B, 16]` |
| Pauli-Z expectations | `[B, 4]` |
| Bias-free up-projection and final descriptor | `[B, 512]` |

Both quantum outputs feed a single bias-free `4 → 512` linear layer. Direct returns the existing L2 normalization of its output. Residual normalizes `x + λΔ(x)` with configurable `λ`, fixed to 0.1 in these experiments. The residual up-projection starts at 0.05 times its seeded linear initialization, so its correction is small and gradients reach the mixer immediately. The epoch trace records the scale and parameter values.

The surrogate replaces the statevector with `tanh(W tanh(z))`, using one bias-free 4-by-4 matrix. Within each Direct or Residual pair, the projections have bitwise identical initial parameters, verified at each seed. The 16 surrogate weights match the trainable quantum angle count. Their initial mixer outputs and functional families differ. A and C retain the existing classical Direct comparison, and D retains the original Residual architecture including its bias. All use the primary `512 → 64 → 4 → 512` layout.

## Verification and five-epoch smoke

- The 42-test suite covered known gates and readouts, deterministic batching, unit normalization, nonzero input and projection gradients, frozen circuit weights, complex128 finite-difference checking, and the training and checkpoint round-trip. It also exercised the original CosPlace checkpoint on synthetic input.
- A two-example cross-check with the installed Qiskit 2.3.1 statevector matched every complex amplitude within `3.44e-16`. With centered differences at step `1e-6`, the largest Jacobian discrepancies were `3.83e-10` for the four inputs and `3.44e-10` for the 16 angles. The adapters themselves use the existing PyTorch dependency.
- All four smoke runs completed five epochs on the full training split at seed 42. Direct sampled losses went from 0.1507 to 0.0799 for quantum and 0.1508 to 0.1018 for the surrogate. Residual began with small loss and its sampled loss fluctuated as the triplets changed. On the fixed set of all eligible training triplets, the smoke-selected checkpoint reduced hinge loss from 0.19533 to 0.11374 for Quantum Direct and from 0.004680 to 0.004628 for Quantum Residual. The corresponding surrogate results were 0.19799 to 0.09868 and 0.004674 to 0.004671. These fixed-set checks used the saved checkpoint and no updates.
- Smoke, benchmark, and ablation evaluations had zero NaN or Inf elements, zero zero-norm descriptors, and zero positive pre-normalization norms below `1e-6`. The largest unit-norm error was `1.92e-7`.

## Runtime and memory

The profile ran before the full benchmark, on macOS ARM64 CPU, PyTorch 2.11.0, one thread. It used the same real training batch of 16 queries and 32 references for each arm, after three warmups. Values are medians of 12 measurements and omit optimizer updates. The archived A/C/D histories did not log their original training wall time. Their checkpoints were profiled in the same batch.

| Model | Inference batch (ms) | Forward + backward batch (ms) |
| --- | ---: | ---: |
| A · Direct ReLU + bias | 0.141 | 0.529 |
| C · Direct LeakyReLU | 0.135 | 0.500 |
| D · Residual MLP | 0.157 | 0.556 |
| Classical surrogate Direct | 0.170 | 0.592 |
| Classical surrogate Residual | 0.201 | 0.696 |
| Quantum Direct | 1.081 | 4.921 |
| Quantum Residual | 1.024 | 4.569 |

The profile process peaked at **290.4 MiB RSS**, including PyTorch and its autograd machinery. One complex64 state tensor for the largest training forward batch is **4 KiB**. This measures feasibility at four wires and batch size 16. Each full quantum run processes 12,000 training and 5,000 validation statevectors across 50 epochs, counting one input descriptor per statevector. Gradients backpropagate through the amplitude operations. Shot sampling is zero because readout uses exact expectations. The timing excludes hardware queues, noise, and measurement costs.

## Controlled validation benchmark

Reused the nine Stage 2.3 primary bottleneck-4 checkpoints for A, C, and D. Checkpoint SHA-256, archived training settings, Recall@K, and MRR all matched. B had three archived zero-descriptor failures at this width. All 12 new primary runs completed using the same 80 training queries, 80 training gallery descriptors, five-frame negative exclusion, Adam at 0.001, margin 0.2, batch size 16, and 50 epochs with patience 50. Each seed uses the same triplet sequence for the matched arms. The earliest lexicographic best validation Recall@1/5/10 selects the checkpoint. Validation ranks the same 50 queries and 50 gallery images, with full-gallery MRR. The test split stayed held out.

Values are mean ± sample SD across seeds 42, 123, and 2026. Recall values are in percent. Time is training wall time including validation and checkpoint writing. The detailed CSV also gives all training split results.

| Model | Complete | Classical / quantum trainable | R@1 | R@5 | R@10 | MRR | Effective rank | Train (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen CosPlace | Fixed | 0 / 0 | 30.00 | 92.00 | 94.00 | 0.5563 | 34.84 | — |
| A · Direct ReLU + bias | 3/3 archived | 35,652 / 0 | 11.33 ± 1.15 | 28.67 ± 2.31 | 45.33 ± 2.31 | 0.2157 ± 0.0123 | 2.23 ± 0.24 | — |
| B · Bias-free ReLU | 0/3 archived | 35,140 / 0 | — | — | — | — | — | — |
| C · Bias-free LeakyReLU | 3/3 archived | 35,140 / 0 | 8.67 ± 1.15 | 26.00 ± 6.00 | 43.33 ± 6.11 | 0.1948 ± 0.0222 | 1.73 ± 0.47 | — |
| D · Residual MLP | 3/3 archived | 35,652 / 0 | 33.33 ± 5.03 | 88.00 ± 2.00 | 94.67 ± 1.15 | 0.5690 ± 0.0171 | 34.62 ± 0.22 | — |
| Classical surrogate Direct | 3/3 | 35,156 / 0 | 9.33 ± 4.16 | 19.33 ± 11.37 | 28.67 ± 16.17 | 0.1688 ± 0.0611 | 1.35 ± 0.06 | 0.33 ± 0.01 |
| Quantum Direct | 3/3 | 35,140 / 16 | 10.67 ± 3.06 | 24.00 ± 5.29 | 40.00 ± 10.58 | 0.1981 ± 0.0397 | 1.52 ± 0.42 | 1.39 ± 0.03 |
| Classical surrogate Residual | 3/3 | 35,156 / 0 | 32.67 ± 2.31 | 92.00 ± 0.00 | 94.00 ± 0.00 | 0.5709 ± 0.0127 | 34.78 ± 0.06 | 0.33 ± 0.00 |
| Quantum Residual | 3/3 | 35,140 / 16 | 33.33 ± 1.15 | 92.00 ± 0.00 | 94.67 ± 1.15 | 0.5723 ± 0.0054 | 34.74 ± 0.21 | 1.35 ± 0.02 |

At the selected validation checkpoints, Quantum Direct has a mean distinct-image cosine of 0.6682 and total variance of 0.3285. Quantum Residual gives 0.1776 and 0.8142. Its initial mean aligned cosine to CosPlace was 0.99977, with a relative correction norm of 0.0214. The selected checkpoint means were 0.99748 and 0.0657. Across wires and seeds, mean validation output SD increased from 0.0248 at initialization to 0.0974 in Quantum Direct and 0.1356 in Quantum Residual. The observed validation encoding saturation fraction was zero. The smallest selected validation raw norm was 0.0747 for Quantum Direct and 0.9758 for Quantum Residual. The per-split CSV contains the full norm summaries and per-wire means and SDs.

Across primary runs, mean batch quantum gradient norms were 0.0848 for Direct and 0.000722 for Residual. Mean population variance across angle-gradient coordinates was 0.00187 and `2.18e-7`, respectively. Mean input-projection gradient norms were 0.2903 and 0.00166. The angle vector moved by a mean L2 distance of 0.2108 and 0.2020 from initialization by the last epoch. Residual gradients are small in the context of a small initial correction and an already low hinge loss. These measurements alone are insufficient to diagnose a barren plateau.

## Targeted ablations

The three ablations each completed the full 50 epochs, changing one factor at seed 42. The residual skip and projection sizes were fixed. “Frozen” keeps the same seeded circuit angles and learns the classical projection layers.

| Quantum Residual variant | Trainable / frozen angles | R@1 | R@5 | R@10 | MRR | Train (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Two layers, CNOT chain | 16 / 0 | 34% | 92% | 94% | 0.5762 | 1.33 |
| One layer | 8 / 0 | 34% | 92% | 96% | 0.5809 | 0.95 |
| Two layers, separable | 16 / 0 | 34% | 90% | 94% | 0.5773 | 1.38 |
| Two layers, frozen angles | 0 / 16 | 34% | 92% | 94% | 0.5762 | 0.88 |

Quantum Direct's mean MRR exceeds its matched surrogate and is below A. Its effective rank is low, consistent with discarding the original descriptor through the narrow bottleneck. Quantum Residual's mean MRR exceeds its matched surrogate by about 0.0014 and was lower for two of the three seeds in the paired comparison. The frozen, shallower, and separable seed-42 results leave the benefit of learning or entangling the circuit unresolved. The surrogate and PQC share parameter count and outer projections, with different initial responses. Selecting checkpoints on one 50-query validation route gives limited evidence about generalization. Repeating the single-seed ablations across seeds and routes would be necessary for stronger attribution. Hardware noise, finite shots, and held-out test retrieval remain unmeasured here.

## Files and reproduction

- [Circuit and encoding](../../src/vpr_baseline/quantum.py), [adapter integration](../../src/vpr_baseline/adapters.py), [gradient monitor](../../src/vpr_baseline/mixer_monitor.py), and [descriptor evaluation](../../src/vpr_baseline/quantum_diagnostics.py).
- [Benchmark detail](results/benchmark.csv), [mean and SD](results/benchmark-summary.csv), [smoke](results/smoke.csv), and [ablations](results/ablations.csv).
- [Batch profile](results/profile.csv), [memory](results/profile-memory.json), and [training trace](results/training-trace.csv) with losses, validation recall, gradient statistics, angle or surrogate weight vectors, and the residual scale per epoch.

Use the existing Gardens Point caches and archived Stage 2.3 checkpoint directories. The profile and benchmark stages verify the original checkpoint hashes. Write each stage to a fresh output directory.

```bash
python -m pip install -e .
export OMP_NUM_THREADS=1
COSPLACE_CHECKPOINT=checkpoints/resnet18_512_cosplace.pth python -m unittest discover -s tests -v

for stage in profile smoke benchmark ablations; do
  python experiments/quantum-stage-3-1/run_stage.py --stage "$stage" \
    --dataset-root data/GardensPointWalking --artifacts-dir artifacts \
    --output-dir "artifacts/quantum-stage-3-1-replay/$stage"
done

vpr-baseline evaluate --dataset-root data/GardensPointWalking \
  --cache-dir artifacts/gardens-point --split val --mrr \
  --adapter-checkpoint artifacts/quantum-stage-3-1-replay/benchmark/runs/quantum_residual-s42/best.pt
```

For individual training use `train-adapter --transform quantum --adapter residual --down-dim 64 --hidden-dim 4 --activation leaky_relu --no-up-bias --alpha 0.1 --epochs 50 --patience 50 --seed 42` with the dataset, cache, and output arguments. `--circuit-depth 1`, `--no-entanglement`, and `--freeze-quantum` expose the ablations. Use `--transform surrogate` with the same projection flags for the classical control.
