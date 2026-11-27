# Classical VPR baseline and adapters

This project extracts frozen CosPlace descriptors with PyTorch and retrieves gallery images by exact cosine similarity. It also trains and evaluates Direct and Residual MLP adapters on cached descriptors. The CLI reports Recall@1, Recall@5, and Recall@10 on validation and test splits.

## Install

The dependency versions are pinned in `pyproject.toml`. The commands below use Python 3.11 and run inference on CPU by default.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
vpr-baseline --help
```

## Get the checkpoint

Download `resnet18_512_cosplace.pth`, the official CosPlace v1.0 ResNet-18 checkpoint. The asset is 45,832,971 bytes (43.7 MiB). The loader checks the SHA-256 below and loads the full state dict strictly. The local architecture uses torchvision's ResNet-18 through the final convolutional stage, followed by CosPlace's channel normalization, learned GeM pooling, linear projection, and descriptor normalization. The model runs with all gradients disabled, in evaluation mode and under PyTorch inference mode. [Official release assets](https://github.com/gmberton/CosPlace/releases/expanded_assets/v1.0) · [Upstream architecture](https://github.com/gmberton/CosPlace/blob/52b56e95ea62245281281f3bafd7b9390d19a0fd/cosplace_model/cosplace_network.py)

```bash
mkdir -p checkpoints
curl --fail --location \
  https://github.com/gmberton/CosPlace/releases/download/v1.0/resnet18_512_cosplace.pth \
  --output checkpoints/resnet18_512_cosplace.pth
shasum -a 256 checkpoints/resnet18_512_cosplace.pth
```

Expected SHA-256

```text
db09310a175e325fe25e7a460b41886ab9357b19c41d370a9ed3ed81bc9abf9f
```

The checkpoint produces **512 values per image**. A missing or changed checkpoint fails before inference. Loading the model needs only this local file. CosPlace code is MIT-licensed and the upstream notice is included. The release page publishes the weights without a separate asset-specific license notice. The weights were trained on SF-XL, which is unnecessary for this baseline. [Upstream project](https://github.com/gmberton/CosPlace) · [CosPlace MIT notice](LICENSES/CosPlace.txt)

## Prepare Gardens Point Walking

Manually download only `GardensPointWalking.zip` (31.6 MB) from *Day and Night, Left and Right* by Arren Glover and Michael Milford, Queensland University of Technology. The Zenodo metadata lists CC BY 4.0 and an archive MD5 of `1284fe9747dd5cb3e6c3358b37ae361f`. Keep the source attribution with any redistribution. The CLI works with local images and does no network I/O. [Dataset record](https://doi.org/10.5281/zenodo.4590133) · [License](https://creativecommons.org/licenses/by/4.0/)

Extract the archive into `data/GardensPointWalking` so that this directory directly contains `day_right/` and `night_right/`. Those folders must contain `Image000.jpg` through `Image199.jpg`.

```bash
mkdir -p data/GardensPointWalking
unzip ~/Downloads/GardensPointWalking.zip -d data/GardensPointWalking
vpr-baseline prepare-gardens-point --dataset-root data/GardensPointWalking
```

The gallery uses the day-right traversal and the queries use night-right. A query's positive is the gallery frame with the same number. This repository's protocol uses contiguous parts of one campus route and leaves ten unused frames at each boundary. Scores describe held-out segments of that route. Record the split and positive definition alongside any scores.

| Split | Frame IDs in each traversal | Images per role |
| --- | --- | ---: |
| `train` | `000–079` | 80 |
| `val` | `090–139` | 50 |
| `test` | `150–199` | 50 |

### Use another small dataset

Create the following UTF-8 CSV manifests under a dataset root. Assign location groups to splits before writing image lists. Keep near-duplicates and all visits to a location in the same split when the dataset allows it. `prepare-gardens-point` creates the same layout for Gardens Point.

```text
dataset_root/
  train/
    gallery.csv
    query.csv
    positives.csv
  val/
    gallery.csv
    query.csv
    positives.csv
  test/
    gallery.csv
    query.csv
    positives.csv
```

Each image CSV uses columns `image_id,path`. Paths are relative to the dataset root and IDs and resolved paths must be unique across the six image manifests. CSV row order defines descriptor row order. Here's a sample gallery row.

```csv
image_id,path
val-g001,day_right/Image090.jpg
```

Each `positives.csv` uses columns `query_id,gallery_id` with one row per relevant pair. The first column must refer to the current split's query manifest and the second to its gallery manifest. Include at least one positive for every query. Training requires `train/positives.csv`. Extraction checks the image manifests for overlap and reads images from the selected split. Evaluation opens the selected split's labels after checking the cache against its manifests. Use validation for experiment selection and reserve test for the final evaluation.

```csv
query_id,gallery_id
val-q001,val-g001
```

## Extract and evaluate

```bash
for split in train val test; do
  vpr-baseline extract --dataset-root data/GardensPointWalking \
    --split "$split" --checkpoint checkpoints/resnet18_512_cosplace.pth \
    --cache-dir artifacts/gardens-point
done

vpr-baseline evaluate --dataset-root data/GardensPointWalking --split val \
  --cache-dir artifacts/gardens-point
vpr-baseline evaluate --dataset-root data/GardensPointWalking --split test \
  --cache-dir artifacts/gardens-point
```

The extractor converts images to RGB, resizes to a fixed 384×512 with antialiased bilinear interpolation, and applies ImageNet channel statistics. A fixed resize supports batches and may change the source aspect ratio. The default is eight images per batch. Set `--height`, `--width`, `--batch-size`, `--workers`, and `--device` to control extraction. Use the same preprocessing for all splits in an experiment.

Each split writes one `artifacts/gardens-point/<split>.npz` archive. It holds `gallery_descriptors` and `query_descriptors` as L2-normalized `float32` matrices, with matching `<role>_ids` and `<role>_paths` arrays. A JSON `metadata` field records the schema version, split, checkpoint hash, descriptor dimension, preprocessing, and software versions. Both roles are published together after extraction succeeds. The archive contains image rows and descriptors only. Read it with `vpr_baseline.cache.load_cache(path)` or `numpy.load(path, allow_pickle=False)`. Evaluate with the archive and the CSVs available locally. Re-extract if the images or preprocessing change.

Retrieval normalizes both matrices and computes their dot products in query batches. Descending scores determine the ranks, with ties resolved in gallery-manifest order. Recall@K is the fraction of all queries with at least one labeled positive in the first `min(K, gallery size)` results. Scores are fractions in `[0, 1]`.

## Train the classical adapters

Both models take a 512-dimensional frozen CosPlace descriptor `z`. The shared MLP uses a linear `512 → 8` projection, a linear `8 → 4` layer followed by ReLU, and a linear `4 → 512` projection. With `delta` as the output of these layers, Direct returns `L2Normalize(delta)` and Residual returns `L2Normalize(z + alpha * delta)`. The same weights transform both gallery and query descriptors within each run. `alpha` is fixed during residual training and defaults to `0.1`.

Use the existing `train.npz` and `val.npz` Phase 1 caches. If you already prepared Gardens Point during Phase 1, rerun `prepare-gardens-point` to add same-frame training labels to `train/positives.csv`. The matching uses frame numbers from `night_right` queries and `day_right` gallery images. Existing descriptor caches can be reused directly. Adapter training and evaluation run on CPU and only require the CSVs and caches.

Each epoch draws one labeled positive and one eligible negative per training query. A gallery frame is eligible only if it is more than `--negative-window` frames away from **every** labeled positive for that query. The window is inclusive at its boundary and uses gallery filenames such as `Image090.jpg` or `090.jpg` as frame numbers. Use a single numbered gallery traversal per split for this policy. Training fails if a query has no positive or no eligible negative. The objective is the mean hinge loss `max(0, margin + cosine(query, negative) - cosine(query, positive))`.

```bash
vpr-baseline prepare-gardens-point --dataset-root data/GardensPointWalking
export OMP_NUM_THREADS=1
for adapter in direct residual; do
  vpr-baseline train-adapter --dataset-root data/GardensPointWalking \
    --cache-dir artifacts/gardens-point --adapter "$adapter" \
    --output-dir "artifacts/gardens-point-adapters/$adapter"
done
```

| Option | Default |
| --- | --- |
| `--down-dim`, `--hidden-dim` | `8`, `4` |
| `--alpha` | `0.1` for Residual |
| `--no-up-bias` | Optional for Direct. The final projection uses a bias by default. |
| `--activation` | `relu` by default. Direct also supports `leaky_relu` with slope `0.01`. |
| `--epochs`, `--patience` | `50`, `10` |
| `--learning-rate`, `--batch-size` | `0.001`, `16` with Adam |
| `--margin`, `--negative-window` | `0.2`, `5` frames |
| `--seed` | `0` |

Use the same seed, cache, widths, and training settings for both variants. The seed controls initialization and triplet sampling. For reproducibility, use the pinned dependencies and the same CPU thread count. Use a new output directory for each run. The command writes `best.pt` and `history.json`, which records the configuration, sampled training loss, validation loss across all eligible validation triplets, and validation Recall@1/5/10 for every epoch.

Only training triplets contribute to parameter updates. Validation Recall@1 selects the checkpoint, followed by Recall@5 and Recall@10 to break ties. Exact ties keep the earlier epoch. Training stops after `--patience` epochs without a better tuple. Review both loss histories alongside recall to monitor overfitting.

### Evaluate and compare

Evaluate a saved checkpoint or compare both adapters and original CosPlace on the same validation labels and gallery. All three use the original cosine ranking, tie handling, and Recall@K implementation. `--output` saves the printed report as JSON.

```bash
vpr-baseline evaluate --dataset-root data/GardensPointWalking \
  --cache-dir artifacts/gardens-point --split val \
  --adapter-checkpoint artifacts/gardens-point-adapters/direct/best.pt

vpr-baseline compare-adapters --dataset-root data/GardensPointWalking \
  --cache-dir artifacts/gardens-point --split val \
  --direct-checkpoint artifacts/gardens-point-adapters/direct/best.pt \
  --residual-checkpoint artifacts/gardens-point-adapters/residual/best.pt \
  --output artifacts/gardens-point-adapters/validation.json
```

Once the configurations and checkpoints are fixed, run the separate test evaluation with the Phase 1 `test.npz` cache.

```bash
vpr-baseline compare-adapters --dataset-root data/GardensPointWalking \
  --cache-dir artifacts/gardens-point --split test \
  --direct-checkpoint artifacts/gardens-point-adapters/direct/best.pt \
  --residual-checkpoint artifacts/gardens-point-adapters/residual/best.pt \
  --output artifacts/gardens-point-adapters/test.json
```

Gardens Point provides 80 training pairs and two 50-query evaluation splits from one campus route. A single query changes Recall@K by two percentage points. The temporal window reduces ambiguity from adjacent frames, and distant frames can still depict visually similar places. Treat single-seed measurements on this protocol as preliminary.

### Adapter diagnostics and bottleneck ablation

The [Task 02.1 experiment report](experiments/classical-bottleneck/README.md) includes train and validation diagnostics, a controlled three-seed bottleneck sweep, CSV results, figures, and reproduction commands.

### Final classical benchmark

The [Task 02.3 report](experiments/classical-bottleneck/final-benchmark/README.md) consolidates all archived classical checkpoints, full-gallery MRR, paired ranks, and the Phase 3 baseline settings. Recreate the tables and figures from the existing cached features and checkpoints.

```bash
python -m pip install -e '.[diagnostics]'
OMP_NUM_THREADS=1 python experiments/classical-bottleneck/run_final_benchmark.py \
  --dataset-root data/GardensPointWalking --artifacts-dir artifacts \
  --output-dir artifacts/final-classical-evaluation
python experiments/classical-bottleneck/plot_final_benchmark.py artifacts/final-classical-evaluation
```

Add `--mrr` to `vpr-baseline evaluate` or `compare-adapters` to report full-gallery MRR alongside Recall@K.

### Stage 3.1 quantum adapters

The [Stage 3.1 report](experiments/quantum-stage-3-1/README.md) documents the differentiable four-qubit circuit, the matched classical surrogate, the controlled train and validation comparison, and commands to reproduce each stage.

## Tests

Run the unit suite and offline CLI smoke tests on generated images, synthetic descriptors, and synthetic labels. The optional Phase 1 real-weight test runs when `COSPLACE_CHECKPOINT` is set.

```bash
python -m unittest discover -s tests -v
COSPLACE_CHECKPOINT=checkpoints/resnet18_512_cosplace.pth \
  python -m unittest discover -s tests -v -k official_checkpoint
```

The real-weight smoke test checks strict loading, 512-element `float32` descriptors, unit norms, and cache-based retrieval with a known match. Its score tests the pipeline on synthetic images.
