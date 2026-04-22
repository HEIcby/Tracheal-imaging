# 推荐参数（仅保留已确认可行的两组）

本文件**只保留两组已确认可行的标志性参数**（`dicom2` 与 `dicom1`）。

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
  "intro": "4/22 dicom2：fixed-threshold=0.33（[0,1]）对比；Z裁剪[-200,-68] start_z=-192.4；section-interval=2。",
  "z_min": -200.0,
  "z_max": -68.0,
  "no_cross_sections": false,
  "section_interval": 2,
  "endoscopy": false,
  "open": false,
  "no_open": true,
  "use_3d_analysis": true,
  "use_propagation": false,
  "start_z": -192.4,
  "start_idx": null,
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
  "intro": "4/22 dicom1：fixed-threshold=0.33（[0,1]）；Z裁剪[-100,50] start_z=-89；section-interval=2。",
  "z_min": -100.0,
  "z_max": 50.0,
  "no_cross_sections": false,
  "section_interval": 2,
  "endoscopy": false,
  "open": false,
  "no_open": true,
  "use_3d_analysis": true,
  "use_propagation": false,
  "start_z": -89.0,
  "start_idx": null,
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
