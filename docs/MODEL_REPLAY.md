# Optional frozen-model inference

The verified default workflow needs only the bundled statistics. This optional layer reruns the 18 primary binary frozen checkpoints on original imagery to regenerate comparator and temperature primitives. It requires external material and a CUDA GPU. The present package verification compiles these sources and checks `--help` in the observed model environment; it does not execute a new GPU replay or retraining.

## Environment and materials

The observed model environment is Python 3.11.16 with torch 2.11.0+cu128, torchvision 0.26.0+cu128 and timm 0.9.2. Full observed package versions are in `model_environment_observed.json`. Use a separate environment from the CPU analysis environment. Install a compatible PyTorch/CUDA build, then the optional `requirements-model.txt`. Rasterio may require its GDAL runtime/data configuration on the target platform. These are observed versions, not a fresh-install portability guarantee.

Create an external data root, for example `external_data/`, outside the frozen `data/` folder. Copy the contents of `model_materials/` there. Copy `data/runs/` to `external_data/runs/`; this supplies the frozen calibration JSON used by inference. Supply the following separately:

```text
external_data/
  configs/                         # from model_materials
  manifests/                       # from model_materials
  sen1floods11/<image_relpath>       # paths from the Sen1Floods11 manifest
  sen1floods11/<label_relpath>
  geobench2/burn_scars/*stats_clip_rescale.json
  geobench2/kuro_siwo/*stats_clip_rescale.json
  third_party/DOFA/                 # compatible original DOFA source
  models/DOFA_ViT_base_e100.pth      # original pretrained base weights
  runs/P1/f1/seed_13/checkpoints/best.pt
  runs/P1/f1/seed_13/risk_v2/calibration/calibration.json
  ...                              # P1/P2 x f1/f2/f3 x seeds 13/37/71
```

The dataset reader initializes all dataset manifests/statistics even for primary replay, so keep all six manifests and both external statistics files. Only Sen1Floods11 imagery is accessed by these primary replay scripts. Full external-model execution additionally needs the original GEO-Bench-2 containers at `geobench2/burn_scars/geobench_burn_scars.tortilla` and `geobench2/kuro_siwo/geobench_kuro_siwo.tortilla` matching the saved byte-offset manifest.

Acquire Sen1Floods11 through its [official repository](https://github.com/cloudtostreet/Sen1Floods11) and external datasets through [GEO-Bench-2](https://github.com/The-AI-Alliance/GEO-Bench-2). Respect their separate terms. Obtain compatible DOFA code from [the official source](https://github.com/zhu-xlab/DOFA); the study used the original `DOFA_ViT_base_e100.pth` from [earthflow/DOFA](https://huggingface.co/earthflow/DOFA). Newer or larger weights are not substitutes for this frozen setup.

`external_material_inventory.json` lists observed checkpoint sizes and SHA-256 fingerprints of the DOFA Python source snapshot. The original DOFA download was a branch ZIP; its commit ID was not recorded, so an exact Git revision is not asserted. The trained study checkpoints are not in this package and have no public download link yet. Consequently, this optional layer is not currently a self-contained end-to-end release.

## Commands

Run one model/partition/seed first, using your environment's Python and your chosen data root:

```powershell
python model_replay/replay_primary_comparators.py --data-root external_data --block P1 --partition F1 --seed 13 --output-dir replay_check/comparator_replay
python model_replay/replay_temperature_ablation.py --data-root external_data --block P1 --partition F1 --seed 13 --output-dir replay_check/temperature_replay
```

Omit `--block`, `--partition` and `--seed` to process all 18 primary runs. Outputs have a CSV and run JSON. The scripts skip a run if both files already exist; choose a new output directory for a genuinely fresh replay. Compare generated primitives, sample order, confusion counts and probabilities with bundled `data/replay/` before using them for inference about results. Completing inference alone does not establish fidelity.

The training and original risk implementations are under `model_code/georisk/`. They are included for method transparency. Historical v1 and v2 configuration choices are preserved; no claim of independent preregistration or deterministic full retraining follows from their presence.
