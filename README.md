# Nadir-NeRF Method + UAV Processing Pipeline

This repository provides:

1. A Nerfstudio method plugin named `nadir-nerf`.
2. A CLI pipeline that automates your MicaSense -> COLMAP(ECEF) -> Nerfstudio workflow.

## Install

```bash
cd nadir-nerf-method
pip install -e ".[uav-pipeline]"
ns-install-cli
```

If your `micasense` package is not pip-installed but exists in source form (for example `../imageprocessing/micasense`), the pipeline will auto-detect it.

## Train Nadir-NeRF (existing ns-data)

```bash
ns-train nadir-nerf --data /path/to/ns-data
```

## End-to-End UAV Pipeline

The new command is:

```bash
nadir-nerf-pipeline run --workspace-dir /path/to/workspace --raw-image-dir /path/to/raw
```

By default, this performs:

1. MicaSense alignment + radiometric output to `workspace/images`
2. `metadata.txt` generation for COLMAP model alignment
3. COLMAP feature extraction, matching, mapping
4. `colmap model_aligner` to ECEF
5. `ns-process-data images --skip-colmap --colmap-model-path colmap/sparse/ecef --use-sfm-depth`

This default path runs fully inside your current active environment.

### Optional train + export in one run

```bash
nadir-nerf-pipeline run \
  --workspace-dir /path/to/workspace \
  --raw-image-dir /path/to/raw \
  --run-train \
  --run-export
```

Pointcloud export defaults:
- `--export-num-points 1000000`
- `--export-remove-outliers true`
- `--export-normal-method open3d`
- `--export-save-world-frame true` (keeps ECEF/world frame)

## Local MicaSense Source (Optional)

If needed, you can point to your source checkout explicitly:

```bash
nadir-nerf-pipeline run \
  --workspace-dir /path/to/workspace \
  --raw-image-dir /path/to/raw \
  --micasense-package-dir /home/myid/ehh67855/thesis/imageprocessing
```

The same flag is available on `nadir-nerf-pipeline preprocess`.

## Useful Stage Controls

- Skip preprocessing: `--skip-preprocess` (requires existing processed images and metadata)
- Skip COLMAP: `--skip-colmap --aligned-model-path /path/to/ecef`
- Skip `ns-process-data`: `--skip-ns-process-data`
- Switch model staging mode: `--colmap-transfer copy|symlink`
- Set alignment tolerance: `--alignment-max-error 10.0`
- Advanced only: use `--preprocess-prefix`, `--colmap-prefix`, `--nerfstudio-prefix` if you still want multi-env execution

## Standalone Preprocess Command

```bash
nadir-nerf-pipeline preprocess \
  --raw-image-dir /path/to/raw \
  --output-image-dir /path/to/images \
  --metadata-path /path/to/metadata.txt
```
