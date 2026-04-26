# 推荐参数（仅保留已确认可行的几组）

本文件**只保留已确认可行的标志性参数**（按样本维护：`dicom2`、`dicom1`、`dicom3`）。

**重要约束**：今后任何对本文件的新增/修改，都必须先征得你的确认。

---

## `dicom2`（已确认可行）

```json
{
  "dicom": ".\\dicom2",
  "output": "trachea_reconstruction",
  "size": 256,
  "iso": 0.5,
  "step": 2,
  "percentile": 36.5,
  "fixed_threshold": 0.33,
  "closing_iters": 2,
  "erosion_iters": 1,
  "intro": "",
  "simple_3d": false,
  "z_min": -200.0,
  "z_max": -68.0,
  "no_cross_sections": false,
  "section_interval": 20,
  "endoscopy": false,
  "open": false,
  "no_open": true,
  "use_3d_analysis": true,
  "use_propagation": false,
  "start_z": -192.4,
  "start_idx": null,
  "navigation_line": true,
  "nav_min_radius": 12.0,
  "expand_trachea": true,
  "expand_threshold": 0.5,
  "expand_max_iters": 200,
  "expand_min_dist_mm": 1.0,
  "expand_by_shell": true,
  "vtk_flythrough": null,
  "_internal": {
    "percentile": 36.5,
    "roi_size": 300,
    "kernel_size": [
      3,
      3
    ],
    "2d_morph": "dilate(iter=1)",
    "3d_closing_iterations": 2,
    "3d_erosion_iterations": 1
  }
}
```

### 视频模式自检（已验证可用）

- 说明：导出 MP4 + `*_sync.json`，并回写注入到 HTML，用于验证“视频模式/严格同步/返回交互对齐”等目标1能力。

```bash
python dicom_trachea_complete.py --dicom .\dicom2 --output trachea_reconstruction --size 256 --iso 0.5 --step 2 --fixed-threshold 0.33 --closing-iters 2 --erosion-iters 1 --z-min -200 --z-max -68 --section-interval 20 --use-3d-analysis --start-z -192.4 --navigation-line --nav-min-radius 12 --expand-by-shell --expand-threshold 0.5 --expand-max-iters 200 --expand-min-dist-mm 1.0 --vtk-flythrough --no-open
```

- 代表性产物（本次已回归验证通过）：
  - `output/20260426_231505/trachea_reconstruction_3d_20260426_231505.html`
  - `output/20260426_231505/trachea_reconstruction_flythrough_20260426_231505_playable.mp4`
  - `output/20260426_231505/trachea_reconstruction_flythrough_20260426_231505_playable_sync.json`

### 多路径算法对比（nav-compare，已验证可用）

```bash
python dicom_trachea_complete.py --dicom .\dicom2 --output trachea_reconstruction --size 256 --iso 0.5 --step 2 --fixed-threshold 0.33 --closing-iters 2 --erosion-iters 1 --z-min -200 --z-max -68 --section-interval 20 --use-3d-analysis --start-z -192.4 --navigation-line --nav-min-radius 12 --nav-compare --expand-by-shell --expand-threshold 0.5 --expand-max-iters 200 --expand-min-dist-mm 1.0 --no-open
```

- 代表性产物（含 `run_args.json/run_metrics.json`）：
  - `output/20260426_225431/run_metrics.json`

---

## `dicom1`（已确认可行）

```json
{
  "dicom": ".\\dicom1",
  "output": "trachea_reconstruction",
  "size": 256,
  "iso": 0.5,
  "step": 2,
  "percentile": 36.5,
  "fixed_threshold": 0.33,
  "closing_iters": 2,
  "erosion_iters": 1,
  "intro": "",
  "simple_3d": false,
  "z_min": -100.0,
  "z_max": 50.0,
  "no_cross_sections": false,
  "section_interval": 15,
  "endoscopy": false,
  "open": false,
  "no_open": true,
  "use_3d_analysis": true,
  "use_propagation": false,
  "start_z": -89.0,
  "start_idx": null,
  "navigation_line": true,
  "nav_min_radius": 12.0,
  "expand_trachea": true,
  "expand_threshold": 0.5,
  "expand_max_iters": 200,
  "expand_min_dist_mm": 0.0,
  "expand_by_shell": true,
  "vtk_flythrough": null,
  "_internal": {
    "percentile": 36.5,
    "roi_size": 300,
    "kernel_size": [
      3,
      3
    ],
    "2d_morph": "dilate(iter=1)",
    "3d_closing_iterations": 2,
    "3d_erosion_iterations": 1
  }
}
```

### 视频模式自检（已验证可用）

```bash
python dicom_trachea_complete.py --dicom .\dicom1 --output trachea_reconstruction --size 256 --iso 0.5 --step 2 --fixed-threshold 0.33 --closing-iters 2 --erosion-iters 1 --z-min -100 --z-max 50 --section-interval 15 --use-3d-analysis --start-z -89 --navigation-line --nav-min-radius 12 --expand-by-shell --expand-max-iters 200 --expand-min-dist-mm 0.0 --vtk-flythrough --no-open
```

- 代表性产物（本次已回归验证通过）：
  - `output/20260426_231205/trachea_reconstruction_3d_20260426_231205.html`
  - `output/20260426_231205/trachea_reconstruction_flythrough_20260426_231205_playable.mp4`
  - `output/20260426_231205/trachea_reconstruction_flythrough_20260426_231205_playable_sync.json`

### 多路径算法对比（nav-compare，已验证可用）

```bash
python dicom_trachea_complete.py --dicom .\dicom1 --output trachea_reconstruction --size 256 --iso 0.5 --step 2 --fixed-threshold 0.33 --closing-iters 2 --erosion-iters 1 --z-min -100 --z-max 50 --section-interval 15 --use-3d-analysis --start-z -89 --navigation-line --nav-min-radius 12 --nav-compare --no-cross-sections --expand-by-shell --expand-max-iters 200 --expand-min-dist-mm 0.0 --no-open
```

- 代表性产物（含 `run_args.json/run_metrics.json`）：
  - `output/20260426_230128/run_metrics.json`

---

## `dicom3`（已确认可行）

```json
{
  "dicom": ".\\dicom3",
  "output": "trachea_reconstruction",
  "size": 512,
  "iso": 0.5,
  "step": 2,
  "percentile": 36.5,
  "fixed_threshold": 0.45,
  "closing_iters": 2,
  "erosion_iters": 1,
  "intro": "",
  "simple_3d": false,
  "z_min": -60.0,
  "z_max": 75.0,
  "no_cross_sections": false,
  "section_interval": 15,
  "endoscopy": false,
  "open": false,
  "no_open": true,
  "use_3d_analysis": true,
  "use_propagation": false,
  "start_z": -50.0,
  "start_idx": null,
  "navigation_line": true,
  "nav_min_radius": 12.0,
  "expand_trachea": true,
  "expand_threshold": 0.5,
  "expand_max_iters": 200,
  "expand_min_dist_mm": 0.0,
  "expand_by_shell": true,
  "vtk_flythrough": null,
  "_internal": {
    "percentile": 36.5,
    "roi_size": 300,
    "kernel_size": [
      3,
      3
    ],
    "2d_morph": "dilate(iter=1)",
    "3d_closing_iterations": 2,
    "3d_erosion_iterations": 1
  }
}
```

### 视频模式自检（已验证可用）

```bash
python dicom_trachea_complete.py --dicom .\dicom3 --output trachea_reconstruction --size 512 --iso 0.5 --step 2 --fixed-threshold 0.45 --closing-iters 2 --erosion-iters 1 --z-min -60 --z-max 75 --section-interval 15 --use-3d-analysis --start-z -50 --navigation-line --nav-min-radius 12 --expand-by-shell --expand-threshold 0.5 --expand-max-iters 200 --expand-min-dist-mm 0.0 --vtk-flythrough --no-open
```

- 代表性产物（本次已回归验证通过）：
  - `output/20260426_231915/trachea_reconstruction_3d_20260426_231915.html`
  - `output/20260426_231915/trachea_reconstruction_flythrough_20260426_231915_playable.mp4`
  - `output/20260426_231915/trachea_reconstruction_flythrough_20260426_231915_playable_sync.json`

### 多路径算法对比（nav-compare，已验证可用）

```bash
python dicom_trachea_complete.py --dicom .\dicom3 --output trachea_reconstruction --size 512 --iso 0.5 --step 2 --fixed-threshold 0.45 --closing-iters 2 --erosion-iters 1 --z-min -60 --z-max 75 --section-interval 15 --use-3d-analysis --start-z -50 --navigation-line --nav-min-radius 12 --nav-compare --no-cross-sections --expand-by-shell --expand-threshold 0.5 --expand-max-iters 200 --expand-min-dist-mm 0.0 --no-open
```

- 代表性产物（含 `run_args.json/run_metrics.json`）：
  - `output/20260426_230210/run_metrics.json`
