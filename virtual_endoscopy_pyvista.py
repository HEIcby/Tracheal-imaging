#!/usr/bin/env python3
"""
充气掩码气管内壁网格 + 路径上的 PyVista 离屏虚拟内镜（射线锥估计前向），导出 MP4。

依赖: pip install pyvista imageio imageio-ffmpeg
与 dicom_trachea_complete 的世界坐标一致: X/Y 为降采样后 mm 量级, Z 为物理 mm。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np


def _reencode_playable_h264(input_mp4: str, output_mp4: str, *, crf: int = 18, fps: Optional[int] = None) -> bool:
    """将可能不兼容的 mp4 重编码为 yuv420p 的 H.264（更通用，便于 Windows 默认播放器播放）。"""
    try:
        import imageio_ffmpeg  # noqa: WPS433
    except Exception:
        return False
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ff,
        "-hide_banner",
        "-y",
        "-i",
        input_mp4,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "baseline",
        "-level",
        "3.1",
        "-movflags",
        "+faststart",
        "-crf",
        str(int(crf)),
        "-preset",
        "medium",
    ]
    if fps is not None:
        cmd += ["-r", str(int(fps))]
    cmd += [output_mp4]
    import subprocess  # noqa: WPS433

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception:
        return False


def _resample_polyline(points: np.ndarray, step_mm: float) -> np.ndarray:
    """沿折线等弧长重采样。"""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return pts
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total < 1e-6:
        return pts
    targets = np.arange(0.0, total, float(step_mm))
    if targets[-1] < total - 1e-3:
        targets = np.append(targets, total)
    out = []
    j = 0
    for t in targets:
        while j < len(s) - 1 and s[j + 1] < t:
            j += 1
        if j >= len(s) - 1:
            out.append(pts[-1].copy())
            continue
        t0, t1 = s[j], s[j + 1]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        out.append((1.0 - u) * pts[j] + u * pts[j + 1])
    return np.array(out, dtype=np.float64)


def _orthonormal_frame(tangent: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(tangent, dtype=np.float64)
    tn = np.linalg.norm(t)
    if tn < 1e-12:
        t = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        t = t / tn
    a = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(t, a)) > 0.92:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(a, t)
    un = np.linalg.norm(u)
    if un < 1e-12:
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        u = u / un
    v = np.cross(t, u)
    v = v / (np.linalg.norm(v) + 1e-12)
    return t, u, v


def _cone_directions(tangent: np.ndarray, semi_deg: float, n_ring: int) -> list:
    t, u, v = _orthonormal_frame(tangent)
    rad = math.radians(semi_deg)
    dirs = [t.copy()]
    for k in range(max(6, n_ring)):
        theta = 2.0 * math.pi * k / max(6, n_ring)
        w = math.cos(rad) * t + math.sin(rad) * (math.cos(theta) * u + math.sin(theta) * v)
        wn = np.linalg.norm(w)
        if wn > 1e-12:
            dirs.append(w / wn)
    return dirs


def _ray_free_length(mesh, origin: np.ndarray, direction: np.ndarray, max_len: float) -> float:
    d = np.asarray(direction, dtype=np.float64)
    d = d / (np.linalg.norm(d) + 1e-12)
    o = np.asarray(origin, dtype=np.float64)
    end = o + d * float(max_len)
    try:
        pts, _ = mesh.ray_trace(o, end, first_point=True)
    except TypeError:
        pts, _ = mesh.ray_trace(o, end)
    if pts is None:
        return float(max_len)
    pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    if pts.shape[0] == 0:
        return float(max_len)
    ts = np.dot(pts - o, d)
    ts = ts[ts > 0.8]
    if ts.size == 0:
        return float(max_len)
    return float(np.min(ts))


def _visibility_forward(
    mesh,
    eye: np.ndarray,
    tangent: np.ndarray,
    *,
    max_ray_mm: float = 90.0,
    semi_deg: float = 34.0,
    n_ring: int = 14,
    blend_tangent: float = 0.42,
) -> np.ndarray:
    t0 = np.asarray(tangent, dtype=np.float64)
    t0 = t0 / (np.linalg.norm(t0) + 1e-12)
    dirs = _cone_directions(t0, semi_deg, n_ring)
    scores = np.array([_ray_free_length(mesh, eye, di, max_ray_mm) for di in dirs], dtype=np.float64)
    w = (scores + 0.5) ** 2
    w = w / (np.sum(w) + 1e-12)
    fwd = sum(w[i] * dirs[i] for i in range(len(dirs)))
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    out = (1.0 - blend_tangent) * fwd + blend_tangent * t0
    return out / (np.linalg.norm(out) + 1e-12)


def _camera_up(forward: np.ndarray, prev_up: Optional[np.ndarray]) -> np.ndarray:
    f = np.asarray(forward, dtype=np.float64)
    f = f / (np.linalg.norm(f) + 1e-12)
    world = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(f, world)) > 0.96:
        world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(f, world)
    rn = np.linalg.norm(right)
    if rn < 1e-8:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / rn
    up = np.cross(right, f)
    up = up / (np.linalg.norm(up) + 1e-12)
    if prev_up is not None:
        pu = np.asarray(prev_up, dtype=np.float64)
        pu = pu / (np.linalg.norm(pu) + 1e-12)
        if np.dot(up, pu) < 0:
            up = -up
        up = 0.72 * up + 0.28 * pu
        up = up / (np.linalg.norm(up) + 1e-12)
    return up


def export_flythrough_mp4(
    verts: np.ndarray,
    faces: np.ndarray,
    path_xyz: np.ndarray,
    output_mp4: str,
    *,
    step_mm: float = 0.8,
    window_size: Tuple[int, int] = (1280, 720),
    fps: int = 24,
    max_ray_mm: float = 85.0,
    focal_min_mm: float = 8.0,
    focal_frac: float = 0.38,
) -> bool:
    """
    verts: (N,3), faces: (M,3) 整型三角形索引。
    path_xyz: (K,3) 相机路径（世界坐标 mm）。
    """
    try:
        import pyvista as pv  # noqa: WPS433
    except ImportError as e:
        print(f"✗ 虚拟内镜视频需要 pyvista: {e}")
        print("  请执行: pip install pyvista imageio imageio-ffmpeg")
        return False

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    path = _resample_polyline(np.asarray(path_xyz, dtype=np.float64), step_mm)
    if len(path) < 3:
        print("✗ 虚拟内镜: 路径点数不足")
        return False

    n_faces = faces.shape[0]
    f_flat = np.empty(n_faces * 4, dtype=np.int64)
    f_flat[0::4] = 3
    f_flat[1::4] = faces[:, 0]
    f_flat[2::4] = faces[:, 1]
    f_flat[3::4] = faces[:, 2]
    surf = pv.PolyData(verts, f_flat)
    surf.compute_normals(
        inplace=True,
        cell_normals=False,
        point_normals=True,
        consistent_normals=True,
        auto_orient_normals=True,
    )

    plotter = pv.Plotter(off_screen=True, window_size=list(window_size))
    plotter.set_background("#05070d")
    try:
        # 更平滑的边缘（不同平台/后端可能不支持，失败则忽略）
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        try:
            plotter.enable_anti_aliasing()
        except Exception:
            pass
    plotter.add_mesh(
        surf,
        color=(0.90, 0.86, 0.80),
        smooth_shading=True,
        specular=0.55,
        specular_power=28.0,
        diffuse=0.92,
        ambient=0.28,
    )

    try:
        # quality: 1(低) ~ 10(高)
        plotter.open_movie(output_mp4, framerate=int(fps), quality=10)
    except Exception as e:
        print(f"✗ 无法创建视频文件（需 imageio-ffmpeg）: {e}")
        plotter.close()
        return False

    prev_up = None
    n = len(path)
    for i in range(n):
        eye = path[i]
        i_prev = max(0, i - 1)
        i_next = min(n - 1, i + 1)
        tangent = path[i_next] - path[i_prev]
        if np.linalg.norm(tangent) < 1e-9:
            tangent = path[min(i + 1, n - 1)] - path[i]
        fwd = _visibility_forward(surf, eye, tangent, max_ray_mm=max_ray_mm)
        scores = []
        for di in _cone_directions(
            fwd / (np.linalg.norm(fwd) + 1e-12), 28.0, 10
        ):
            scores.append(_ray_free_length(surf, eye, di, max_ray_mm))
        med_free = float(np.median(scores)) if scores else max_ray_mm * 0.35
        focal_dist = float(np.clip(focal_frac * med_free, focal_min_mm, 55.0))
        focal = eye + fwd * focal_dist
        up = _camera_up(fwd, prev_up)
        prev_up = up
        plotter.camera.position = eye.tolist()
        plotter.camera.focal_point = focal.tolist()
        plotter.camera.up = up.tolist()
        plotter.camera.clipping_range = (0.1, 5000.0)
        plotter.write_frame()

    plotter.close()
    print(f"✓ PyVista 虚拟内镜 MP4 已保存: {output_mp4}")

    # 某些播放器无法播放 High 4:4:4 Predictive 等编码配置；额外输出一个更通用的版本
    playable = None
    if output_mp4.lower().endswith(".mp4"):
        playable = output_mp4[:-4] + "_playable.mp4"
    if playable and playable != output_mp4:
        if _reencode_playable_h264(output_mp4, playable, crf=18, fps=int(fps)):
            print(f"✓ 兼容版 MP4 已保存: {playable}")
        else:
            print("⚠ 未能生成兼容版 MP4（可忽略；如无法播放请手动用 ffmpeg 转码为 yuv420p）")
    return True


def export_from_pipeline(pipeline, output_mp4: str, **kwargs) -> bool:
    """从 DicomTrachea3DPipeline 实例导出（需已完成充气法网格与路径）。"""
    verts = getattr(pipeline, "trachea_lumen_verts", None)
    faces = getattr(pipeline, "trachea_lumen_faces", None)
    if verts is None or faces is None:
        print("✗ 虚拟内镜: 无气管内壁网格（需 --use-3d-analysis 且充气法网格生成成功）")
        return False
    nav = getattr(pipeline, "navigation_path_plotly", None)
    cl = getattr(pipeline, "centerline_world", None)
    if nav is not None and len(nav) > 3:
        path = np.asarray(nav, dtype=np.float64)
    elif cl is not None and len(cl) > 3:
        path = np.asarray(cl, dtype=np.float64)
    else:
        print("✗ 虚拟内镜: 无可用路径（需中心线或导航线）")
        return False
    return export_flythrough_mp4(verts, faces, path, output_mp4, **kwargs)
