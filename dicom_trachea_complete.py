#!/usr/bin/env python3
"""
DICOM完整3D重建 + 气管中心线提取
集成40%分位算法的精确中心线标注

完整流程:
1. DICOM → 按ImagePositionPatient排序
2. 提取气管中心线(40%分位+连通域分析)
3. Marching Cubes生成3D网格
4. Plotly可视化(网格+中心线+虚拟内窥镜)

作者: 集成analyze_dicom_improved.py的优化算法
"""

import numpy as np
import pydicom
import cv2
from skimage import measure
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import argparse
import glob
import os
import sys
import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image
import webbrowser

try:
    # Windows 默认控制台可能是 gbk，含 Emoji/中文时容易炸；强制 UTF-8 更稳
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


class DicomTrachea3DPipeline:
    """DICOM气管3D重建 + 中心线提取完整流程"""
    
    def __init__(
        self,
        dicom_dir,
        output_name="trachea_reconstruction",
        *,
        percentile=36.5,
        closing_iters=2,
        erosion_iters=1,
        fixed_threshold=None,
    ):
        self.dicom_dir = dicom_dir
        self.output_name = output_name
        
        # 算法参数(来自analyze_dicom_improved.py)
        self.percentile = float(percentile)  # 分位阈值（越小越保守，区域越窄；支持小数）
        self.roi_size = 300         # ROI大小
        self.area_min = 30          # 最小面积（只过滤噪声）
        self.area_max = float('inf') # 不限制上限
        self.kernel_size = (3, 3)   # 形态学核
        self.closing_iters = int(closing_iters)
        self.erosion_iters = int(erosion_iters)
        # 固定阈值（用于3D充气法二值化）。取值建议为窗位窗宽归一化后的[0,1]。
        self.fixed_threshold = None if fixed_threshold is None else float(fixed_threshold)
        
        # 数据
        self.slices_data = []       # [(z_pos, ds, pixel_array), ...]
        self.volume = None
        self.vertices = None
        self.faces = None
        self.downsample_size = None  # 记录降采样尺寸
        self.original_size = None    # 记录原始尺寸
        self.cross_section_analysis = []  # 保存横截面分析结果
        # 运行/实验信息（由main注入，用于写入HTML底部）
        self.experiment_intro = ""
        self.experiment_args = {}
        self.experiment_started_at = None  # datetime.datetime
        # 导航线（从3D掩码计算；用于插管路径/相机漫游）
        self.navigation_path_plotly = None  # (N,3): x_ds, y_ds, z_mm
        self.navigation_path_voxel = None   # (N,3): z_idx, y_roi, x_roi（ROI原始分辨率）
        self.navigation_meta = {}

    def _estimate_spacing_mm(self):
        """估计体素间距（mm）。X/Y来自PixelSpacing，Z用切片物理Z的中位差。"""
        if not self.slices_data:
            return 1.0, 1.0, 1.0

        ds0 = self.slices_data[0][1]
        row_mm, col_mm = 1.0, 1.0
        try:
            if hasattr(ds0, "PixelSpacing") and len(ds0.PixelSpacing) >= 2:
                row_mm = float(ds0.PixelSpacing[0])
                col_mm = float(ds0.PixelSpacing[1])
        except Exception:
            pass

        z_positions = [float(z) for z, _, _ in self.slices_data]
        if len(z_positions) >= 2:
            dzs = np.abs(np.diff(np.array(z_positions, dtype=np.float64)))
            dz_mm = float(np.median(dzs)) if np.any(dzs > 0) else 1.0
        else:
            dz_mm = 1.0

        return dz_mm, row_mm, col_mm

    def _polyline_min_turn_radius_mm(self, pts_mm: np.ndarray):
        """返回折线的最小转弯半径(mm)。三点外接圆半径作为离散曲率近似。"""
        if pts_mm is None or len(pts_mm) < 3:
            return float("inf")
        p = pts_mm.astype(np.float64)
        min_r = float("inf")
        for i in range(1, len(p) - 1):
            a = p[i - 1]
            b = p[i]
            c = p[i + 1]
            ab = b - a
            bc = c - b
            ac = c - a
            lab = np.linalg.norm(ab)
            lbc = np.linalg.norm(bc)
            lac = np.linalg.norm(ac)
            if lab < 1e-6 or lbc < 1e-6 or lac < 1e-6:
                continue
            # 三角形面积 = 0.5*|ab x ac|
            area2 = np.linalg.norm(np.cross(ab, ac))
            if area2 < 1e-9:
                continue
            r = (lab * lbc * lac) / (area2 + 1e-12)
            if r < min_r:
                min_r = r
        return min_r

    def _resample_polyline_by_step(self, pts_mm: np.ndarray, step_mm: float):
        """按弧长步长重采样折线（线性插值）。"""
        if pts_mm is None or len(pts_mm) < 2:
            return pts_mm
        p = pts_mm.astype(np.float64)
        seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
        total = float(np.sum(seg))
        if total <= 1e-6:
            return p
        step_mm = max(0.5, float(step_mm))
        n = max(2, int(np.floor(total / step_mm)) + 1)
        t = np.concatenate([[0.0], np.cumsum(seg)])
        t_new = np.linspace(0.0, t[-1], n)
        out = np.zeros((n, 3), dtype=np.float64)
        j = 0
        for i, ti in enumerate(t_new):
            while j < len(t) - 2 and ti > t[j + 1]:
                j += 1
            t0, t1 = t[j], t[j + 1]
            if t1 <= t0 + 1e-9:
                out[i] = p[j]
            else:
                alpha = (ti - t0) / (t1 - t0)
                out[i] = (1 - alpha) * p[j] + alpha * p[j + 1]
        return out

    def _find_nearest_skeleton_voxel(self, target_zyx, skel: np.ndarray, max_radius=25):
        """在骨架体素中找离target最近的点（逐步扩张立方邻域；避免全局DT的大内存）。"""
        tz, ty, tx = (int(target_zyx[0]), int(target_zyx[1]), int(target_zyx[2]))
        z_max, y_max, x_max = skel.shape
        if 0 <= tz < z_max and 0 <= ty < y_max and 0 <= tx < x_max and skel[tz, ty, tx]:
            return (tz, ty, tx)
        for r in range(1, int(max_radius) + 1):
            z0, z1 = max(0, tz - r), min(z_max - 1, tz + r)
            y0, y1 = max(0, ty - r), min(y_max - 1, ty + r)
            x0, x1 = max(0, tx - r), min(x_max - 1, tx + r)
            window = skel[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1]
            if not np.any(window):
                continue
            coords = np.argwhere(window)
            coords[:, 0] += z0
            coords[:, 1] += y0
            coords[:, 2] += x0
            d2 = np.sum((coords - np.array([tz, ty, tx], dtype=np.int32)) ** 2, axis=1)
            best = coords[int(np.argmin(d2))]
            return (int(best[0]), int(best[1]), int(best[2]))
        return None

    def _compute_navigation_path_from_mask(
        self,
        *,
        min_turn_radius_mm: float = 12.0,
        resample_step_mm: float = 1.0,
        wall_bias_power: float = 1.5,
        max_smoothing_iters: int = 6,
    ):
        """
        从3D二值掩码计算“管腔中心导航线”。
        - 仅使用体素图 + 几何代价（不参考2D中心线点列）
        - 在掩码内部偏好“远离壁”的路径：基于距离变换的代价
        输出：
          - self.navigation_path_voxel: (N,3) z_idx,y_roi,x_roi（ROI原始分辨率）
          - self.navigation_path_plotly: (N,3) x_ds,y_ds,z_mm（与HTML一致坐标系）
        """
        if self.trachea_mask_3d is None or self.slices_data is None or len(self.slices_data) == 0:
            return False

        mask = (self.trachea_mask_3d > 0)
        if mask.ndim != 3 or np.sum(mask) < 50:
            return False

        dz_mm, dy_mm, dx_mm = self._estimate_spacing_mm()

        # 距离变换（mm）：在掩码内部，离壁越远值越大
        from scipy.ndimage import distance_transform_edt
        dist_mm = distance_transform_edt(mask, sampling=(dz_mm, dy_mm, dx_mm)).astype(np.float32)

        # 骨架（细化）作为可搜索的中心候选集
        print("\n计算导航线：3D骨架化 + 路径搜索...")
        try:
            # scikit-image < 0.20
            from skimage.morphology import skeletonize_3d as _skel3d  # type: ignore
            skel = _skel3d(mask).astype(bool)
        except Exception:
            try:
                # scikit-image >= 0.20：skeletonize 支持 nD（含 3D）
                from skimage.morphology import skeletonize
                skel = skeletonize(mask.astype(bool)).astype(bool)
            except Exception as e:
                print(f"  ✗ 3D骨架化不可用: {e}")
                return False
        skel_count = int(np.sum(skel))
        print(f"  骨架体素: {skel_count:,}")
        if skel_count < 50:
            print("  ✗ 骨架体素过少，跳过导航线")
            return False

        # 起终点：沿堆叠方向，从“物理Z较大端(zmax/头侧)”与“物理Z较小端(zmin/足侧)”
        # 各自向内找到第一个仍有掩码的切片（充气结果常在两端若干层为空，不能硬用0/N-1）
        zN = mask.shape[0]

        def pick_best_voxel_at_slice(z_idx: int):
            m = mask[z_idx]
            if not np.any(m):
                return None
            d = dist_mm[z_idx].copy()
            d[~m] = -1.0
            yx = np.unravel_index(int(np.argmax(d)), d.shape)
            return (z_idx, int(yx[0]), int(yx[1]))

        z_start = None
        for z_idx in range(zN - 1, -1, -1):
            if np.any(mask[z_idx]):
                z_start = int(z_idx)
                break
        z_goal = None
        for z_idx in range(0, zN):
            if np.any(mask[z_idx]):
                z_goal = int(z_idx)
                break

        if z_start is None or z_goal is None:
            print("  ✗ 掩码在所有切片上为空，无法设置起终点")
            return False

        start_guess = pick_best_voxel_at_slice(z_start)
        goal_guess = pick_best_voxel_at_slice(z_goal)
        if start_guess is None or goal_guess is None:
            print("  ✗ 起点或终点切片没有有效掩码体素")
            return False

        start = self._find_nearest_skeleton_voxel(start_guess, skel) or start_guess
        goal = self._find_nearest_skeleton_voxel(goal_guess, skel) or goal_guess

        # Dijkstra on skeleton voxels
        import heapq
        zN, yN, xN = mask.shape

        def lin(zyx):
            z, y, x = zyx
            return (z * yN + y) * xN + x

        start_l = lin(start)
        goal_l = lin(goal)

        offsets = []
        for oz in (-1, 0, 1):
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    if oz == 0 and oy == 0 and ox == 0:
                        continue
                    offsets.append((oz, oy, ox))

        def step_len_mm(oz, oy, ox):
            return float(np.sqrt((oz * dz_mm) ** 2 + (oy * dy_mm) ** 2 + (ox * dx_mm) ** 2))

        heap = [(0.0, start_l)]
        dist = {start_l: 0.0}
        parent = {start_l: None}
        visited = 0

        while heap:
            cur_d, cur_l = heapq.heappop(heap)
            if cur_l == goal_l:
                break
            if cur_d != dist.get(cur_l, None):
                continue
            visited += 1
            # 解码
            z = cur_l // (yN * xN)
            rem = cur_l - z * (yN * xN)
            y = rem // xN
            x = rem - y * xN
            for oz, oy, ox in offsets:
                nz, ny, nx = z + oz, y + oy, x + ox
                if not (0 <= nz < zN and 0 <= ny < yN and 0 <= nx < xN):
                    continue
                if not skel[nz, ny, nx]:
                    continue
                wall_d = float(dist_mm[nz, ny, nx])
                # dist很小时给大惩罚，迫使路径远离壁
                cost = step_len_mm(oz, oy, ox) * (1.0 / (wall_d + 1e-3)) ** float(wall_bias_power)
                nl = (nz * yN + ny) * xN + nx
                nd = cur_d + cost
                if nd < dist.get(nl, float("inf")):
                    dist[nl] = nd
                    parent[nl] = cur_l
                    heapq.heappush(heap, (nd, nl))

        if goal_l not in parent:
            print("  ✗ 骨架上未找到连通路径（可能骨架断裂）")
            return False

        # 回溯路径
        path_l = []
        cur = goal_l
        while cur is not None:
            path_l.append(cur)
            cur = parent[cur]
        path_l.reverse()

        path_zyx = np.zeros((len(path_l), 3), dtype=np.int32)
        for i, ll in enumerate(path_l):
            z = ll // (yN * xN)
            rem = ll - z * (yN * xN)
            y = rem // xN
            x = rem - y * xN
            path_zyx[i] = (z, y, x)

        print(f"  骨架路径长度: {len(path_zyx)}点（visited={visited:,}）")

        # 转为物理坐标(mm)用于平滑与转弯半径约束
        # ROI内像素坐标 -> 原始全局像素坐标 -> mm
        x1_orig, y1_orig = self.roi_offset  # 原始分辨率ROI偏移
        row_mm, col_mm = dy_mm, dx_mm
        pts_mm = np.zeros((len(path_zyx), 3), dtype=np.float64)
        for i, (z_idx, y_roi, x_roi) in enumerate(path_zyx):
            z_mm = float(self.slices_data[int(z_idx)][0])
            x_orig = float(int(x_roi) + int(x1_orig))
            y_orig = float(int(y_roi) + int(y1_orig))
            pts_mm[i] = (x_orig * col_mm, y_orig * row_mm, z_mm)

        # 重采样 + 样条平滑，尽量满足最小转弯半径
        pts_mm = self._resample_polyline_by_step(pts_mm, resample_step_mm)
        best_mm = pts_mm
        best_r = self._polyline_min_turn_radius_mm(best_mm)

        if len(pts_mm) >= 4:
            for it in range(max_smoothing_iters):
                # splprep需要参数t，使用弧长参数化更稳定
                try:
                    # s从小到大递增：更平滑 -> 更大转弯半径（但可能偏离中心）
                    s_val = float((it + 1) * 5.0)
                    tck, _ = splprep([pts_mm[:, 0], pts_mm[:, 1], pts_mm[:, 2]], s=s_val, k=3)
                    u = np.linspace(0.0, 1.0, len(pts_mm))
                    x_s, y_s, z_s = splev(u, tck)
                    sm = np.stack([x_s, y_s, z_s], axis=1)
                    sm = self._resample_polyline_by_step(sm, resample_step_mm)
                    r = self._polyline_min_turn_radius_mm(sm)
                    if r > best_r:
                        best_r = r
                        best_mm = sm
                    if r >= float(min_turn_radius_mm):
                        best_mm = sm
                        best_r = r
                        break
                except Exception:
                    break

        # 沿路径索引再轻平滑：削弱骨架/样条在局部段的左右摆动，减轻漫游时“抖出管壁感”
        try:
            from scipy.ndimage import gaussian_filter1d
            sig = 1.15
            for d in range(3):
                best_mm[:, d] = gaussian_filter1d(best_mm[:, d], sigma=sig, mode="nearest")
            best_mm = self._resample_polyline_by_step(best_mm, resample_step_mm)
            best_r = self._polyline_min_turn_radius_mm(best_mm)
        except Exception:
            pass

        print(f"  最小转弯半径(估计): {best_r:.1f} mm（目标≥{float(min_turn_radius_mm):.1f} mm）")

        # 将平滑后的(mm)路径回写到 Plotly 坐标系（x_ds,y_ds,z_mm）并保存体素索引近似
        # 这里做“近似回投影”：用mm反算原始像素坐标，再减去ROI偏移得到y_roi/x_roi
        scale_factor = self.downsample_size / self.original_size[0]
        x_ds = (best_mm[:, 0] / col_mm) * scale_factor
        y_ds = (best_mm[:, 1] / row_mm) * scale_factor
        z_mm_arr = best_mm[:, 2]
        self.navigation_path_plotly = np.stack([x_ds, y_ds, z_mm_arr], axis=1).astype(np.float32)

        # voxel路径用于调试/后续映射：z_idx用“最近切片”匹配
        z_positions = np.array([float(z) for z, _, _ in self.slices_data], dtype=np.float64)
        voxel_path = np.zeros((len(best_mm), 3), dtype=np.int32)
        for i in range(len(best_mm)):
            z_idx = int(np.argmin(np.abs(z_positions - float(best_mm[i, 2]))))
            x_orig = int(round(float(best_mm[i, 0]) / col_mm))
            y_orig = int(round(float(best_mm[i, 1]) / row_mm))
            voxel_path[i, 0] = z_idx
            voxel_path[i, 1] = int(y_orig - int(y1_orig))
            voxel_path[i, 2] = int(x_orig - int(x1_orig))
        self.navigation_path_voxel = voxel_path

        self.navigation_meta = {
            "min_turn_radius_mm_target": float(min_turn_radius_mm),
            "min_turn_radius_mm_estimated": float(best_r if np.isfinite(best_r) else 0.0),
            "resample_step_mm": float(resample_step_mm),
            "wall_bias_power": float(wall_bias_power),
            "spacing_mm": {"dz": float(dz_mm), "dy": float(dy_mm), "dx": float(dx_mm)},
            "direction": "zmax_to_zmin",
        }

        return True
    
    def step1_load_and_sort_dicom(self, z_min=None, z_max=None):
        """步骤1: 加载DICOM并按ImagePositionPatient的Z坐标排序"""
        print("\n" + "="*60)
        print("步骤 1/4: 加载DICOM文件并按空间位置排序")
        print("="*60)
        
        # 查找DICOM文件
        dicom_files = []
        dicom_root = Path(self.dicom_dir)
        for ext in ['*.dcm', '*.DCM', '*.dicom', '*.DICOM']:
            dicom_files.extend(str(p) for p in dicom_root.rglob(ext) if p.is_file())
        
        if not dicom_files:
            # 兼容无扩展名/嵌套目录（例如 dicom2/SE5/IM0）
            for p in dicom_root.rglob('*'):
                if not p.is_file():
                    continue
                try:
                    pydicom.dcmread(str(p), stop_before_pixels=True)
                    dicom_files.append(str(p))
                except Exception:
                    pass
        
        print(f"找到 {len(dicom_files)} 个DICOM文件")
        
        # 按ImagePositionPatient的Z坐标排序(关键!)
        slices = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath)
                if hasattr(ds, 'ImagePositionPatient') and hasattr(ds, 'pixel_array'):
                    z_pos = float(ds.ImagePositionPatient[2])
                    
                    if z_min is not None and z_pos < z_min:
                        continue
                    if z_max is not None and z_pos > z_max:
                        continue
                    
                    slices.append((z_pos, ds, ds.pixel_array))
            except Exception as e:
                continue
        
        # 按Z坐标排序
        slices.sort(key=lambda x: x[0])
        self.slices_data = slices
        
        # 记录原始图像尺寸
        if slices:
            self.original_size = slices[0][2].shape  # (height, width)
        
        print(f"✓ 有效切片: {len(slices)}")
        if slices:
            print(f"  Z轴范围: {slices[0][0]:.1f}mm 到 {slices[-1][0]:.1f}mm")
        
        return len(slices)
    
    def _generate_cross_section_analysis(self, z_idx, slice_data, slice_hu=None, mask_3d=None, separated_regions=None):
        """
        生成单个横截面的详细分析图像
        返回: dict包含各步骤的图像base64编码
        
        参数:
            mask_3d: 3D充气法预计算的mask（可选）
            separated_regions: 3D分析识别的分离区域列表（可选）
        
        ⭐ 关键: 和step2中心线提取一样,只在中心ROI区域分析
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        analysis = {
            'z_index': z_idx,
            'images': {}
        }
        
        # 获取图像尺寸和中心
        height, width = slice_data.shape
        center_y, center_x = height // 2, width // 2
        
        # ⭐ 关键修复: 提取中心ROI (和analyze_dicom_improved.py一致)
        roi_size = int(self.roi_size * self.downsample_size / self.original_size[0])  # 缩放到降采样尺寸
        half_roi = roi_size // 2
        
        y1 = max(0, center_y - half_roi)
        y2 = min(height, center_y + half_roi)
        x1 = max(0, center_x - half_roi)
        x2 = min(width, center_x + half_roi)
        
        roi = slice_data[y1:y2, x1:x2]
        
        # 1. 原始切片(显示ROI框)
        if slice_hu is not None:
            # 在原始HU图像上显示ROI框
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(slice_hu, cmap='gray', vmin=-1000, vmax=500)
            
            # 计算原始尺寸的ROI位置
            scale_factor = self.original_size[0] / self.downsample_size
            roi_y1 = int(y1 * scale_factor)
            roi_y2 = int(y2 * scale_factor)
            roi_x1 = int(x1 * scale_factor)
            roi_x2 = int(x2 * scale_factor)
            
            rect = plt.Rectangle((roi_x1, roi_y1), roi_x2-roi_x1, roi_y2-roi_y1,
                                fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.set_title(f'原始切片 Z={z_idx}\nROI: {roi_x2-roi_x1}×{roi_y2-roi_y1}', fontsize=10)
            ax.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            analysis['images']['original'] = base64.b64encode(buf.read()).decode()
            plt.close()
        
        # 2. 阈值二值化 (⭐ 只在ROI上操作)
        # 与3D充气法保持一致：fixed-threshold 模式使用窗位窗宽归一化后的[0,1]阈值；否则使用分位阈值
        non_bg = roi[roi > 0.1]
        if len(non_bg) >= 100:
            if self.fixed_threshold is not None:
                threshold = float(self.fixed_threshold)
                roi_windowed = roi  # slice_data 已是[0,1]，这里保持兼容命名
                binary = ((roi > 0.1) & (roi_windowed < threshold)).astype(np.uint8)
                binary_title = f'二值化 (固定阈值={threshold:g})\n阈值域=[0,1]'
            else:
                threshold = np.percentile(non_bg, self.percentile)
                binary = ((roi > 0.1) & (roi < threshold)).astype(np.uint8)
                binary_title = f'二值化 ({self.percentile:g}%分位)\nROI内阈值={threshold:.3f}'
            
            # 边缘清零
            binary[0, :] = 0
            binary[-1, :] = 0
            binary[:, 0] = 0
            binary[:, -1] = 0
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(binary, cmap='gray')
            ax.set_title(binary_title, fontsize=10)
            ax.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            analysis['images']['binary'] = base64.b64encode(buf.read()).decode()
            plt.close()
            
            # 3. 形态学操作后（只膨胀，扩散白色区域）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary_morph = cv2.dilate(binary, kernel, iterations=1)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(binary_morph, cmap='gray')
            ax.set_title('形态学操作\n(膨胀扩散)', fontsize=10)
            ax.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            analysis['images']['morphology'] = base64.b64encode(buf.read()).decode()
            plt.close()
            
            # 4. 连通域标记
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_morph, connectivity=8
            )
            
            # 彩色标签
            labels_colored = np.zeros((*labels.shape, 3), dtype=np.uint8)
            np.random.seed(42)
            colors = np.random.randint(50, 255, size=(num_labels, 3), dtype=np.uint8)
            for label_id in range(1, num_labels):
                mask = labels == label_id
                labels_colored[mask] = colors[label_id]
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(labels_colored)
            ax.set_title(f'连通域标记\n共{num_labels-1}个', fontsize=10)
            ax.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            analysis['images']['connected_components'] = base64.b64encode(buf.read()).decode()
            plt.close()
            
            # 4.5 3D连通性合并（在面积筛选之前）
            if mask_3d is not None and separated_regions is not None:
                num_separated = len(separated_regions)
                roi_normalized = ((roi - roi.min()) / (roi.max() - roi.min() + 1e-8) * 255).astype(np.uint8)
                merge_img = cv2.cvtColor(roi_normalized, cv2.COLOR_GRAY2BGR)
                
                if num_separated > 1:
                    # 用不同颜色标记每个分离的2D区域（同一切面内）
                    merge_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
                    num_labels_3d, labels_3d, stats_3d, centroids_3d = cv2.connectedComponentsWithStats(mask_3d, connectivity=8)
                    
                    for i, region in enumerate(separated_regions):
                        color = merge_colors[i % len(merge_colors)]
                        lid = region['id']
                        region_mask = (labels_3d == lid).astype(np.uint8)
                        contours_temp, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours_temp:
                            cv2.drawContours(merge_img, contours_temp, -1, color, 2)
                            cx, cy = int(region['center'][0]), int(region['center'][1])
                            cv2.putText(merge_img, f"{i+1}", (cx-5, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 画连接线表示这些区域在当前切面内被合并
                    if len(separated_regions) >= 2:
                        for i in range(len(separated_regions) - 1):
                            c1 = separated_regions[i]['center']
                            c2 = separated_regions[i+1]['center']
                            cv2.line(merge_img, (int(c1[0]), int(c1[1])), (int(c2[0]), int(c2[1])), 
                                    (255, 255, 255), 1, cv2.LINE_AA)
                    
                    fig, ax = plt.subplots(figsize=(4, 4.5))
                    ax.imshow(cv2.cvtColor(merge_img, cv2.COLOR_BGR2RGB))
                    ax.set_title(f'4.5 同切面3D合并\n本层{num_separated}个分离域→1个', fontsize=9)
                    ax.axis('off')
                    info_text = f"本切面2D分离\n但3D连通→合并为1"
                    ax.text(0.5, -0.02, info_text, transform=ax.transAxes, fontsize=7, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    analysis['images']['merge_3d'] = base64.b64encode(buf.read()).decode()
                    plt.close()
                    
                    # ⭐ 用3D mask替换，让后续步骤使用合并后的结果
                    binary_morph = mask_3d.copy()
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_morph, connectivity=8)
                else:
                    contours_merged, _ = cv2.findContours(mask_3d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours_merged:
                        cv2.drawContours(merge_img, contours_merged, -1, (0, 255, 0), 2)
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.imshow(cv2.cvtColor(merge_img, cv2.COLOR_BGR2RGB))
                    ax.set_title('4.5 3D连通性\n气管完整', fontsize=9)
                    ax.axis('off')
                    buf = BytesIO()
                    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                    buf.seek(0)
                    analysis['images']['merge_3d'] = base64.b64encode(buf.read()).decode()
                    plt.close()
                    binary_morph = mask_3d.copy()
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_morph, connectivity=8)
            
            # 5. 面积筛选
            area_min = 30
            # ⭐ 纯2D路径时过滤体外大空气区（>200px），充气法路径不限上限（已由3D验证）
            area_max = float('inf') if (mask_3d is not None) else 200
            
            filtered_mask = np.zeros_like(binary_morph, dtype=np.uint8)
            candidates = []
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area_min <= area <= area_max:
                    filtered_mask[labels == label_id] = 255
                    candidates.append(label_id)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(filtered_mask, cmap='gray')
            area_max_str = f'<={int(area_max)}px' if area_max != float('inf') else '不限'
            ax.set_title(f'面积筛选\n{len(candidates)}/{num_labels-1}个\n({area_min}~{area_max_str})', fontsize=9)
            ax.axis('off')
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            analysis['images']['area_filter'] = base64.b64encode(buf.read()).decode()
            plt.close()
            
            # 6. 圆形度评分 (在ROI上显示)
            score_img = ((roi - roi.min()) / (roi.max() - roi.min() + 1e-6) * 255).astype(np.uint8)
            score_img = cv2.cvtColor(score_img, cv2.COLOR_GRAY2BGR)
            
            # ⭐ 优先使用中心线投影作为锚点，回退到图像中心
            anchor_global_x = None
            anchor_global_y = None
            anchor_label = 'CENTERLINE'
            
            if hasattr(self, 'centerline_roi') and self.centerline_roi is not None and len(self.centerline_roi) > 0:
                centerline_roi_array = self.centerline_roi
                # ⭐ 顺序追踪：使用严格在当前切片之前的最近有效切片质心
                prev_mask = centerline_roi_array[:, 0] < z_idx
                if np.any(prev_mask):
                    prev_indices = np.where(prev_mask)[0]
                    nearest_prev_idx = prev_indices[-1]  # z < z_idx 中最近的
                    anchor_global_y = float(centerline_roi_array[nearest_prev_idx, 1]) + y1
                    anchor_global_x = float(centerline_roi_array[nearest_prev_idx, 2]) + x1
                else:
                    # 没有前一个切片（当前是最靠前的），取最近的（可能是后面的切片）
                    z_diffs = np.abs(centerline_roi_array[:, 0] - z_idx)
                    nearest_idx = np.argmin(z_diffs)
                    anchor_global_y = float(centerline_roi_array[nearest_idx, 1]) + y1
                    anchor_global_x = float(centerline_roi_array[nearest_idx, 2]) + x1
                    anchor_label = 'CENTERLINE\n(no prev)'
            
            if anchor_global_x is None or anchor_global_y is None:
                # 回退：使用固定图像中心
                anchor_global_x = self.original_size[1] / 2 * (self.downsample_size / self.original_size[1])
                anchor_global_y = self.original_size[0] / 2 * (self.downsample_size / self.original_size[0])
                anchor_label = 'CENTER\n(fallback)'
            
            candidates_info = []  # (label_id, dist_to_anchor)
            
            for label_id in candidates:
                mask = (labels == label_id).astype(np.uint8)
                contours_temp, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours_temp:
                    continue
                
                area = stats[label_id, cv2.CC_STAT_AREA]
                
                # 只要通过了面积筛选,就都是候选
                cx_roi, cy_roi = centroids[label_id]  # ROI局部坐标 (centroids返回[x,y])
                # 转换到降采样后的全局坐标，计算距离中心线锚点的距离
                global_cy_downsample = cy_roi + y1
                global_cx_downsample = cx_roi + x1
                dist = np.sqrt((global_cx_downsample - anchor_global_x)**2 + (global_cy_downsample - anchor_global_y)**2)
                candidates_info.append((label_id, dist))
            
            # 选择策略：选择距离中心线锚点最近的候选
            if len(candidates_info) == 1:
                best_idx = candidates_info[0][0]
            elif len(candidates_info) > 1:
                best_candidate = min(candidates_info, key=lambda x: x[1])
                best_idx = best_candidate[0]
                best_score = best_candidate[1]
            else:
                best_idx = -1
                best_score = 0
            
            # 6. 距离筛选 - 可视化距离计算
            # 使用ROI归一化图像作为背景
            roi_normalized = ((roi - roi.min()) / (roi.max() - roi.min() + 1e-8) * 255).astype(np.uint8)
            distance_img = cv2.cvtColor(roi_normalized, cv2.COLOR_GRAY2BGR)
            
            # 计算锚点在ROI中的局部位置
            center_roi_x = anchor_global_x - x1
            center_roi_y = anchor_global_y - y1
            
            # 标记锚点（大红色十字）
            if 0 <= center_roi_x < width and 0 <= center_roi_y < height:
                cv2.drawMarker(distance_img, (int(center_roi_x), int(center_roi_y)), 
                              (0, 0, 255), cv2.MARKER_CROSS, 20, 3)
                cv2.putText(distance_img, 'CL', 
                           (int(center_roi_x) + 8, int(center_roi_y) - 8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 绘制所有候选及其到中心线锚点的连线
            for label_id, dist in candidates_info:
                mask = (labels == label_id).astype(np.uint8)
                contours_temp, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # 候选质心 (ROI局部坐标)
                cx_roi, cy_roi = centroids[label_id]  # centroids返回[x,y]
                
                # 绘制候选轮廓
                if label_id == best_idx:
                    color = (0, 255, 0)  # 绿色 - 最佳
                    line_thickness = 2
                else:
                    color = (100, 100, 100)  # 灰色 - 其他
                    line_thickness = 1
                
                cv2.drawContours(distance_img, [contours_temp[0]], -1, color, 2)
                
                # 画从锚点到质心的连线 (都使用ROI局部坐标)
                if 0 <= center_roi_x < width and 0 <= center_roi_y < height:
                    cv2.line(distance_img, 
                            (int(center_roi_x), int(center_roi_y)),
                            (int(cx_roi), int(cy_roi)),
                            color, line_thickness, cv2.LINE_AA)
                
                # 标记候选质心（小圆点）
                cv2.circle(distance_img, (int(cx_roi), int(cy_roi)), 3, color, -1)
            
            # 准备距离信息文本（按距离排序）
            sorted_candidates = sorted(candidates_info, key=lambda x: x[1])
            distance_text = f"Distance to {anchor_label.split(chr(10))[0]}:\n"
            for label_id, dist in sorted_candidates:
                marker = "✓" if label_id == best_idx else " "
                distance_text += f"{marker} ID{label_id}: {dist:.0f}px\n"
            
            fig, ax = plt.subplots(figsize=(4, 4.5))
            ax.imshow(distance_img)
            title_text = '6. 距离中心线筛选\n绿色=最近'
            ax.set_title(title_text, fontsize=9)
            ax.axis('off')
            
            # 在图片下方添加距离信息
            ax.text(0.5, -0.05, distance_text.strip(), 
                   transform=ax.transAxes,
                   fontsize=7, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontfamily=['SimHei', 'Microsoft YaHei', 'monospace'])
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            analysis['images']['distance'] = base64.b64encode(buf.read()).decode()
            plt.close()
            
            # 7. 最终结果（保持原有的评分图）
            # 绘制所有候选(灰色)和最佳候选(绿色)
            for label_id, dist in candidates_info:
                mask = (labels == label_id).astype(np.uint8)
                contours_temp, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                color = (0, 255, 0) if label_id == best_idx else (100, 100, 100)
                cv2.drawContours(score_img, [contours_temp[0]], -1, color, 2)
            
            # 准备距离信息文本（按距离排序）
            sorted_candidates = sorted(candidates_info, key=lambda x: x[1])
            distance_text = f"候选距离{anchor_label.split(chr(10))[0]}:\n"
            for label_id, dist in sorted_candidates:
                marker = "✓" if label_id == best_idx else " "
                distance_text += f"{marker} ID{label_id}: {dist:.0f}px\n"
            
            fig, ax = plt.subplots(figsize=(4, 4.5))
            ax.imshow(score_img)
            title_text = '距离中心线评分\n绿色=最佳'
            ax.set_title(title_text, fontsize=9)
            ax.axis('off')
            
            # 在图片下方添加距离信息
            ax.text(0.5, -0.05, distance_text.strip(), 
                   transform=ax.transAxes,
                   fontsize=7, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontfamily=['SimHei', 'Microsoft YaHei', 'monospace'])
            
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            analysis['images']['scoring'] = base64.b64encode(buf.read()).decode()
            plt.close()
            
            # 7. 最终结果
            if best_idx != -1:
                result_img = cv2.cvtColor(score_img, cv2.COLOR_BGR2RGB)
                
                mask = (labels == best_idx).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    cv2.drawContours(result_img, contours, -1, (255, 255, 0), 3)
                    
                    # 使用距离变换计算中心点(最大内切圆的中心)
                    from scipy.ndimage import distance_transform_edt
                    mask_best = (labels == best_idx).astype(np.uint8)
                    dist_transform = distance_transform_edt(mask_best)
                    
                    # 找到最大距离的位置(中心点)
                    center_coords = np.unravel_index(dist_transform.argmax(), dist_transform.shape)
                    cy_roi = center_coords[0]  # Y坐标
                    cx_roi = center_coords[1]  # X坐标
                    
                    # 在ROI图像上标注选中连通域的中心点(红色十字)
                    cv2.drawMarker(result_img, (int(cx_roi), int(cy_roi)), (255, 0, 0),
                                  cv2.MARKER_CROSS, 10, 2)
                    
                    # 标注图像中心点(256,256)在ROI中的位置(青色十字)
                    # 图像中心在降采样空间中的坐标
                    image_center_downsample_x = self.original_size[1] / 2 * (self.downsample_size / self.original_size[1])
                    image_center_downsample_y = self.original_size[0] / 2 * (self.downsample_size / self.original_size[0])
                    # 转换到ROI局部坐标
                    image_center_roi_x = image_center_downsample_x - x1
                    image_center_roi_y = image_center_downsample_y - y1
                    # 如果图像中心在ROI范围内，则标注
                    if 0 <= image_center_roi_x < width and 0 <= image_center_roi_y < height:
                        cv2.drawMarker(result_img, (int(image_center_roi_x), int(image_center_roi_y)), 
                                      (0, 255, 255), cv2.MARKER_CROSS, 15, 3)  # 青色，稍大
                        cv2.putText(result_img, 'IMG_CENTER', 
                                   (int(image_center_roi_x) + 10, int(image_center_roi_y) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                    
                    # 计算全局坐标用于统计
                    cx_global_downsample = cx_roi + x1
                    cy_global_downsample = cy_roi + y1
                    cx_original = cx_global_downsample * (self.original_size[1] / self.downsample_size)
                    cy_original = cy_global_downsample * (self.original_size[0] / self.downsample_size)
                    
                    area = stats[best_idx, cv2.CC_STAT_AREA]
                    perimeter = cv2.arcLength(contours[0], True)
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    
                    # 保存轮廓点坐标（ROI局部坐标）和ROI偏移
                    contour_points_roi = contours[0][:, 0, :].copy()  # [N, 2] 形状
                    
                    analysis['stats'] = {
                        'center': (float(cx_original), float(cy_original)),  # 返回原始坐标
                        'center_roi': (float(cx_roi), float(cy_roi)),  # 也保存ROI坐标用于标注
                        'contour_roi': contour_points_roi.tolist(),  # 保存轮廓点（ROI坐标）
                        'roi_offset': (int(x1), int(y1)),  # ROI在降采样空间中的偏移
                        'area': int(area)
                    }
                
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.imshow(result_img)
                ax.set_title('最终结果\n黄色=轮廓, 红色=中心', fontsize=10)
                ax.axis('off')
                
                if 'stats' in analysis:
                    info_text = (f"中心: ({analysis['stats']['center'][0]:.1f}, {analysis['stats']['center'][1]:.1f})\n"
                               f"面积: {analysis['stats']['area']}px")
                    ax.text(0.5, -0.05, info_text, transform=ax.transAxes,
                           fontsize=8, ha='center', va='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
                buf.seek(0)
                analysis['images']['final'] = base64.b64encode(buf.read()).decode()
                plt.close()
        
        return analysis
    
    def _extract_trachea_3d_volume(self, start_z_physical=-100.0, start_z_idx=None, end_z_physical=None):
        """
        从3D体数据中提取气管体积
        
        策略:
        1. 从每个切片提取候选连通域(不强制圆形)
        2. 从Z=-100mm附近找起始切片
        3. 选择起始切片中最靠近中心的候选
        4. 向头侧和足侧传播,基于质心连续性和面积连续性
        5. 检测异常切片
        
        参数:
            start_z_physical: 起始物理Z坐标(mm),默认-100.0
            end_z_physical: 结束物理Z坐标(mm),默认None(使用全部切片)
        
        返回:
            selected_labels: {z_idx: (label_id, centroid, area), ...}
        """
        print("\n" + "="*60)
        print("步骤: 提取气管3D体积(基于连通域传播)")
        print("="*60)
        
        # 1. 提取所有切片的候选连通域（使用滑动窗口自适应面积阈值）
        print("1. 提取所有切片的候选连通域...")
        
        candidates_per_slice = {}  # {z_idx: [(label_id, centroid, area, mask), ...]}
        
        # 滑动窗口面积历史（用于自适应阈值）
        area_history_init = []
        window_size_init = 10  # 初始阶段使用较大窗口
        
        for z_idx in range(len(self.slices_data)):
            z_pos, ds, pixel_array = self.slices_data[z_idx]
            
            # 如果指定了end_z_physical,跳过超出范围的切片
            if end_z_physical is not None and z_pos > end_z_physical:
                continue
            
            # 转换为HU值
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                hu_image = pixel_array.astype(np.float32) * slope + intercept
            else:
                hu_image = pixel_array.astype(np.float32)
            
            # 提取ROI
            height, width = hu_image.shape
            center_y, center_x = height // 2, width // 2
            half_roi = self.roi_size // 2
            y1 = max(0, center_y - half_roi)
            y2 = min(height, center_y + half_roi)
            x1 = max(0, center_x - half_roi)
            x2 = min(width, center_x + half_roi)
            roi = hu_image[y1:y2, x1:x2]
            
            # 二值化(分位阈值 - 只选择最黑的气管区域)
            non_bg = roi[roi > -1900]
            if len(non_bg) < 100:
                candidates_per_slice[z_idx] = []
                continue
            
            threshold = np.percentile(non_bg, self.percentile)
            binary = ((roi > -1900) & (roi < threshold)).astype(np.uint8)
            
            # 边缘清零
            binary[0, :] = 0
            binary[-1, :] = 0
            binary[:, 0] = 0
            binary[:, -1] = 0
            
            # 形态学操作（只膨胀，扩散白色区域）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
            binary = cv2.dilate(binary, kernel, iterations=1)
            
            # 连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=8
            )
            
            if num_labels <= 1:
                candidates_per_slice[z_idx] = []
                continue
            
            # ===== 智能预合并：合并因堵塞物分裂的连通域 =====
            if num_labels > 2:  # 至少有2个候选才需要合并
                roi_h, roi_w = roi.shape
                roi_center = (roi_w / 2, roi_h / 2)
                
                # 1. 收集中心区域的所有连通域信息
                center_components = []
                for lid in range(1, num_labels):
                    area = stats[lid, cv2.CC_STAT_AREA]
                    if area < 30:  # 过滤极小噪声
                        continue
                    cx, cy = centroids[lid]
                    dist_to_center = np.sqrt((cx - roi_center[0])**2 + (cy - roi_center[1])**2)
                    
                    # 只考虑中心40%区域内的连通域
                    if dist_to_center < roi_h * 0.4:
                        # 计算该区域的平均HU值
                        comp_mask = (labels == lid)
                        comp_hu = np.mean(roi[comp_mask]) if np.sum(comp_mask) > 0 else -1000
                        
                        center_components.append({
                            'id': lid,
                            'area': area,
                            'centroid': (cx, cy),
                            'dist_to_center': dist_to_center,
                            'mean_hu': comp_hu
                        })
                
                # 2. 按距离中心排序
                if len(center_components) >= 2:
                    center_components.sort(key=lambda x: x['dist_to_center'])
                    
                    # 3. 计算预期面积
                    expected_area = np.mean(area_history_init[-window_size_init:]) if len(area_history_init) >= 3 else 500
                    
                    # 4. 检查是否需要合并（最靠近中心的连通域面积偏小）
                    main_comp = center_components[0]
                    if main_comp['area'] < expected_area * 0.6:
                        # 计算合并距离阈值（基于预期半径的2倍）
                        merge_radius = np.sqrt(expected_area / np.pi) * 2.5
                        
                        merge_list = [main_comp]
                        merged_area = main_comp['area']
                        
                        for comp in center_components[1:]:
                            # 计算到主连通域的距离
                            dist = np.sqrt((comp['centroid'][0] - main_comp['centroid'][0])**2 +
                                          (comp['centroid'][1] - main_comp['centroid'][1])**2)
                            
                            # 合并条件：距离近、HU值相似（都是空气）、合并后面积合理
                            hu_similar = abs(comp['mean_hu'] - main_comp['mean_hu']) < 300
                            area_ok = merged_area + comp['area'] < expected_area * 3.0
                            
                            if dist < merge_radius and hu_similar and area_ok:
                                merge_list.append(comp)
                                merged_area += comp['area']
                        
                        # 5. 执行合并
                        if len(merge_list) > 1:
                            new_label = merge_list[0]['id']
                            merged_ids = [c['id'] for c in merge_list]
                            
                            # 将所有待合并的连通域标记为同一个ID
                            for comp in merge_list[1:]:
                                labels[labels == comp['id']] = new_label
                            
                            # 更新stats
                            stats[new_label, cv2.CC_STAT_AREA] = int(merged_area)
                            
                            # 更新centroids（加权平均）
                            total_area = sum(c['area'] for c in merge_list)
                            new_cx = sum(c['centroid'][0] * c['area'] for c in merge_list) / total_area
                            new_cy = sum(c['centroid'][1] * c['area'] for c in merge_list) / total_area
                            centroids[new_label] = (new_cx, new_cy)
                            
                            if z_idx % 20 == 0:  # 每20个切片打印一次
                                print(f"    z_idx={z_idx}: 🔗预合并 {len(merge_list)}个连通域 "
                                      f"IDs={merged_ids}, 面积:{main_comp['area']}->{int(merged_area)}px")
            
            # 自适应面积阈值（基于滑动窗口）
            scale_factor = (height / self.original_size[0]) ** 2
            
            if len(area_history_init) >= 3:
                # 使用滑动窗口计算自适应阈值
                # 原则：只过滤极小噪声
                avg_area = np.mean(area_history_init[-window_size_init:])
                area_min_adaptive = max(30, avg_area * 0.2)  # 只过滤噪声
                area_max_adaptive = float('inf')  # 上限不限制
            else:
                # 初始阶段使用宽松的固定阈值
                area_min_adaptive = 30  # 只过滤噪声
                area_max_adaptive = float('inf')  # 上限不限制
            
            # 提取所有候选(只需面积合理,不强制圆形)
            slice_candidates = []
            slice_valid_areas = []  # 收集有效面积用于更新滑动窗口
            
            # ⭐ 使用中心线投影作为锚点（而非固定图像中心）
            centerline_anchor_x = None
            centerline_anchor_y = None
            
            if hasattr(self, 'centerline_roi') and self.centerline_roi is not None and len(self.centerline_roi) > 0:
                # 找到当前切片对应的中心线点
                centerline_roi_array = self.centerline_roi
                slice_centerline = centerline_roi_array[centerline_roi_array[:, 0] == z_idx]
                
                if len(slice_centerline) > 0:
                    centerline_anchor_y_roi, centerline_anchor_x_roi = slice_centerline[0, 1], slice_centerline[0, 2]
                    centerline_anchor_y = centerline_anchor_y_roi + y1
                    centerline_anchor_x = centerline_anchor_x_roi + x1
                else:
                    # 如果没有精确匹配，使用最近的切片
                    z_diffs = np.abs(centerline_roi_array[:, 0] - z_idx)
                    nearest_idx = np.argmin(z_diffs)
                    centerline_anchor_y_roi, centerline_anchor_x_roi = centerline_roi_array[nearest_idx, 1], centerline_roi_array[nearest_idx, 2]
                    centerline_anchor_y = centerline_anchor_y_roi + y1
                    centerline_anchor_x = centerline_anchor_x_roi + x1
            
            # 如果没有中心线数据，回退到图像中心
            if centerline_anchor_x is None or centerline_anchor_y is None:
                centerline_anchor_y = self.original_size[0] / 2
                centerline_anchor_x = self.original_size[1] / 2
            
            for label_id in range(1, num_labels):
                area = stats[label_id, cv2.CC_STAT_AREA]
                
                # 自适应面积过滤
                if area < area_min_adaptive or area > area_max_adaptive:
                    continue
                
                # 计算圆形度(用于记录,但不强制过滤)
                mask = (labels == label_id).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if not contours:
                    continue
                
                # 质心(ROI坐标)
                cx, cy = centroids[label_id]  # centroids返回[x,y]
                
                # 转换为全局坐标
                global_cx = cx + x1
                global_cy = cy + y1
                
                # HU值辅助判断：计算该区域的平均HU值
                # 正常气管内HU约-900~-1000，堵塞物HU > -500
                region_hu_values = roi[mask > 0]
                mean_hu = np.mean(region_hu_values) if len(region_hu_values) > 0 else -1000
                is_obstructed = mean_hu > -500  # 堵塞标志
                
                slice_candidates.append((
                    label_id, 
                    (global_cx, global_cy), 
                    area, 
                    mask,
                    mean_hu,        # 新增：平均HU值
                    is_obstructed   # 新增：是否堵塞
                ))
                
                # 收集有效面积用于更新滑动窗口
                slice_valid_areas.append(area)
            
            # 更新滑动窗口面积历史（选择最接近中心线的候选面积）
            if slice_candidates:
                # ⭐ 找最靠近中心线投影的候选
                center_dists = [np.sqrt((c[1][0] - centerline_anchor_x)**2 + (c[1][1] - centerline_anchor_y)**2) 
                               for c in slice_candidates]
                best_idx = np.argmin(center_dists)
                area_history_init.append(slice_candidates[best_idx][2])
                
                # 保持窗口大小
                if len(area_history_init) > window_size_init * 2:
                    area_history_init = area_history_init[-window_size_init * 2:]
            
            candidates_per_slice[z_idx] = slice_candidates
        
        print(f"  提取了 {len(candidates_per_slice)} 个切片的候选")
        
        # 2. 找到最接近start_z_physical的起始切片（或直接使用start_z_idx）
        if start_z_idx is not None:
            # 直接使用指定的z_idx
            if start_z_idx < 0 or start_z_idx >= len(self.slices_data):
                print(f"  ✗ 指定的z_idx={start_z_idx}超出范围!")
                return {}
            if not candidates_per_slice.get(start_z_idx, []):
                print(f"  ✗ 指定的z_idx={start_z_idx}没有候选!")
                return {}
            start_z_pos = self.slices_data[start_z_idx][0]
            print(f"\n2. 使用指定起始切片...")
            print(f"  起始切片: z_idx={start_z_idx}, Z={start_z_pos:.1f}mm")
        else:
            # 通过物理坐标查找
            print(f"\n2. 寻找起始切片(目标Z≈{start_z_physical}mm)...")
            
            start_z_idx = None
            min_z_diff = float('inf')
            
            for z_idx in range(len(self.slices_data)):
                z_pos = self.slices_data[z_idx][0]
                if candidates_per_slice.get(z_idx, []):
                    z_diff = abs(z_pos - start_z_physical)
                    if z_diff < min_z_diff:
                        min_z_diff = z_diff
                        start_z_idx = z_idx
            
            if start_z_idx is None:
                print("  ✗ 未找到有效的起始切片!")
                return {}
            
            start_z_pos = self.slices_data[start_z_idx][0]
            print(f"  起始切片: z_idx={start_z_idx}, Z={start_z_pos:.1f}mm (误差={min_z_diff:.1f}mm)")
        
        # 3. 在起始切片中选择最优候选（使用圆形度+距离综合评分）
        start_candidates = candidates_per_slice[start_z_idx]
        height, width = self.slices_data[start_z_idx][2].shape
        img_center = (width / 2, height / 2)
        
        # 为起始切片计算圆形度（仅用于起始选择）
        start_candidates_with_circ = []
        for cand in start_candidates:
            label_id, centroid, area, mask = cand[:4]
            
            # 计算圆形度
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                else:
                    circularity = 0
            else:
                circularity = 0
            
            # 计算距离中心
            dist = np.sqrt((centroid[0] - img_center[0])**2 + (centroid[1] - img_center[1])**2)
            
            # 综合评分：圆形度权重0.6，距离权重0.4（距离归一化到0-1，越近越好）
            max_dist = np.sqrt(width**2 + height**2) / 2
            dist_score = 1.0 - (dist / max_dist)  # 归一化，越近分数越高
            score = circularity * 0.6 + dist_score * 0.4
            
            start_candidates_with_circ.append((label_id, centroid, area, mask, circularity, dist, score))
        
        # 选择得分最高的
        best_candidate = max(start_candidates_with_circ, key=lambda c: c[6])  # c[6]是score
        
        print(f"  起始候选: label={best_candidate[0]}, 中心={best_candidate[1]}, "
              f"面积={best_candidate[2]}, 圆形度={best_candidate[4]:.3f}, 距离={best_candidate[5]:.1f}px")
        
        # 4. 向两个方向传播
        print("\n3. 向头侧和足侧传播...")
        
        selected_labels = {start_z_idx: best_candidate[:3]}  # 存储(label_id, centroid, area)
        
        def propagate_selection(z_range, direction_name):
            """传播选择函数 - 增强版：自适应合并、质心平滑、预测追踪"""
            prev_z_idx = start_z_idx
            prev_centroid = best_candidate[1]
            prev_area = best_candidate[2]
            skip_count = 0  # 连续跳过计数
            max_skip = 5    # 增加到5个切片（改进3：跳跃恢复）
            
            # 改进4：质心轨迹平滑 - 使用滑动窗口
            centroid_history = [prev_centroid]  # 保存最近5个质心
            area_history = [prev_area]          # 保存最近5个面积
            window_size = 5
            
            # 改进5：预测式追踪 - 记录速度向量
            velocity = (0.0, 0.0)  # 质心移动速度
            
            for z_idx in z_range:
                all_candidates = candidates_per_slice.get(z_idx, [])
                
                # 改进2：自适应面积阈值 - 基于滑动窗口平均
                # 原则：只过滤极小噪声
                avg_area = np.mean(area_history[-window_size:]) if area_history else prev_area
                area_min_dynamic = max(30, avg_area * 0.2)  # 只过滤噪声
                area_max_dynamic = float('inf')    # 上限不限制
                candidates = [c for c in all_candidates 
                             if area_min_dynamic <= c[2]]  # 只检查下限
                
                if not candidates:
                    skip_count += 1
                    if skip_count <= max_skip:
                        if z_idx % 10 == 0 or skip_count == 1:
                            print(f"    {direction_name} z_idx={z_idx}: 无候选 [跳过 {skip_count}/{max_skip}]")
                        continue
                    else:
                        # 改进3：跳跃恢复机制 - 向前探测10个切片
                        print(f"    {direction_name} z_idx={z_idx}: 尝试跳跃恢复...")
                        recovery_found = False
                        probe_range = list(z_range)[list(z_range).index(z_idx):list(z_range).index(z_idx)+10] if z_idx in list(z_range) else []
                        
                        for probe_z in probe_range[1:]:  # 跳过当前，探测后面
                            probe_candidates = candidates_per_slice.get(probe_z, [])
                            # 使用预测位置筛选
                            predicted_centroid = (
                                prev_centroid[0] + velocity[0] * (probe_z - prev_z_idx),
                                prev_centroid[1] + velocity[1] * (probe_z - prev_z_idx)
                            )
                            
                            for cand in probe_candidates:
                                cand_dist = np.sqrt(
                                    (cand[1][0] - predicted_centroid[0])**2 + 
                                    (cand[1][1] - predicted_centroid[1])**2
                                )
                                cand_area_ratio = cand[2] / avg_area
                                
                                # 放宽条件：距离<80px 且 面积在合理范围
                                if cand_dist < 80 and 0.3 <= cand_area_ratio <= 3.0:
                                    print(f"    {direction_name} 🔄 跳跃恢复成功! z_idx={probe_z}, "
                                          f"距离预测={cand_dist:.1f}px")
                                    # 跳到恢复点继续
                                    z_idx = probe_z
                                    candidates = [cand]
                                    skip_count = 0
                                    recovery_found = True
                                    break
                            if recovery_found:
                                break
                        
                        if not recovery_found:
                            print(f"    {direction_name}传播终止: 连续{max_skip}个切片无候选且恢复失败")
                            break
                
                # 改进4：使用平滑后的质心作为参考
                if len(centroid_history) >= 2:
                    # 加权平均，越近的权重越大
                    weights = np.exp(np.linspace(-1, 0, min(len(centroid_history), window_size)))
                    weights = weights / weights.sum()
                    recent = centroid_history[-window_size:]
                    smoothed_centroid = (
                        sum(w * c[0] for w, c in zip(weights[-len(recent):], recent)),
                        sum(w * c[1] for w, c in zip(weights[-len(recent):], recent))
                    )
                else:
                    smoothed_centroid = prev_centroid
                
                # 改进5：计算预测位置
                predicted_centroid = (
                    smoothed_centroid[0] + velocity[0],
                    smoothed_centroid[1] + velocity[1]
                )
                
                # 计算每个候选的综合得分（考虑预测位置和HU值）
                scores = []
                for cand in candidates:
                    # 到平滑质心的距离
                    centroid_dist = np.sqrt(
                        (cand[1][0] - smoothed_centroid[0])**2 + 
                        (cand[1][1] - smoothed_centroid[1])**2
                    )
                    
                    # 到预测位置的距离
                    predict_dist = np.sqrt(
                        (cand[1][0] - predicted_centroid[0])**2 + 
                        (cand[1][1] - predicted_centroid[1])**2
                    )
                    
                    # 面积比
                    area_ratio = cand[2] / avg_area if avg_area > 0 else 1.0
                    area_change = abs(area_ratio - 1.0)
                    
                    # 改进1: HU值惩罚 - 堵塞区域(HU > -500)得分增加
                    hu_penalty = 0.0
                    if len(cand) > 4:  # 有HU值信息
                        mean_hu = cand[4]
                        is_obstructed = cand[5] if len(cand) > 5 else False
                        # 正常气管HU约-900~-1000，堵塞物HU > -500
                        if is_obstructed or mean_hu > -500:
                            hu_penalty = 50  # 显著惩罚堵塞区域
                        elif mean_hu > -700:
                            hu_penalty = 20  # 轻微惩罚可疑区域
                    
                    # 综合得分: 质心距离0.35 + 预测距离0.25 + 面积变化0.25 + HU惩罚0.15
                    score = (centroid_dist * 0.35 + predict_dist * 0.25 + 
                            area_change * 100 * 0.25 + hu_penalty * 0.15)
                    scores.append(score)
                
                # 选择得分最低的候选
                best_idx = np.argmin(scores)
                best_score = scores[best_idx]
                best_cand = candidates[best_idx]
                
                # 改进2：自适应多连通域合并策略
                merged_candidates = [best_cand]
                merged_area = best_cand[2]
                merged_label_ids = [best_cand[0]]
                
                # 自适应合并距离阈值：基于历史平均半径的1.5倍
                avg_radius = np.sqrt(avg_area / np.pi) if avg_area > 0 else 30
                merge_dist_threshold = avg_radius * 1.5
                
                # 如果最佳候选面积 < 60% 预期面积，尝试合并
                if merged_area < avg_area * 0.6:
                    for idx, cand in enumerate(candidates):
                        if idx == best_idx:
                            continue
                        
                        # 计算与最佳候选的距离
                        dist_to_best = np.sqrt(
                            (cand[1][0] - best_cand[1][0])**2 + 
                            (cand[1][1] - best_cand[1][1])**2
                        )
                        
                        # 使用自适应距离阈值
                        if dist_to_best < merge_dist_threshold:
                            merged_candidates.append(cand)
                            merged_area += cand[2]
                            merged_label_ids.append(cand[0])
                    
                    # 如果合并后面积更合理，使用合并结果
                    if len(merged_candidates) > 1 and merged_area >= avg_area * 0.5:
                        # 重新计算合并后的质心（加权平均）
                        total_area = sum(c[2] for c in merged_candidates)
                        merged_cx = sum(c[1][0] * c[2] for c in merged_candidates) / total_area
                        merged_cy = sum(c[1][1] * c[2] for c in merged_candidates) / total_area
                        merged_centroid = (merged_cx, merged_cy)
                        
                        # 更新best_cand为合并后的结果
                        best_cand = (merged_label_ids, merged_centroid, merged_area)
                        
                        print(f"    {direction_name} z_idx={z_idx}: 🔧合并{len(merged_candidates)}个连通域 "
                              f"labels={merged_label_ids}, 总面积={merged_area:.0f}px")
                
                # 计算距离和面积比（使用合并后的数据）
                centroid_dist = np.sqrt(
                    (best_cand[1][0] - prev_centroid[0])**2 + 
                    (best_cand[1][1] - prev_centroid[1])**2
                )
                area_ratio = best_cand[2] / prev_area
                
                # 检测异常跳跃 - 尝试主动修复
                is_abnormal = False
                reason = ""
                
                if centroid_dist > 50:  # 质心距离阈值(像素)
                    is_abnormal = True
                    reason = f"质心跳跃({centroid_dist:.1f}px)"
                elif area_ratio < 0.4 or area_ratio > 2.5:  # 面积变化阈值
                    is_abnormal = True
                    reason = f"面积变化({area_ratio:.2f})"
                
                if is_abnormal:
                    # 🔧 主动修复：尝试从所有候选中找到质心跳跃更小的
                    repair_candidates = []
                    for idx, cand in enumerate(candidates):
                        if idx == best_idx:  # 跳过已检测为异常的最佳候选
                            continue
                        repair_dist = np.sqrt(
                            (cand[1][0] - prev_centroid[0])**2 + 
                            (cand[1][1] - prev_centroid[1])**2
                        )
                        repair_area_ratio = cand[2] / prev_area
                        
                        # 检查是否满足正常条件
                        if repair_dist <= 50 and 0.4 <= repair_area_ratio <= 2.5:
                            repair_candidates.append((idx, cand, repair_dist, repair_area_ratio))
                    
                    if repair_candidates:
                        # 找到了可修复的候选，选择质心距离最小的
                        repair_candidates.sort(key=lambda x: x[2])  # 按质心距离排序
                        repair_idx, repair_cand, repair_dist, repair_area_ratio = repair_candidates[0]
                        
                        print(f"    {direction_name} z_idx={z_idx}: {reason} 🔧已修复 -> "
                              f"label={repair_cand[0]}, 质心距离={repair_dist:.1f}px, 面积比={repair_area_ratio:.2f}")
                        
                        # 使用修复后的候选
                        best_cand = repair_cand
                        centroid_dist = repair_dist
                        area_ratio = repair_area_ratio
                        skip_count = 0  # 重置跳过计数
                    else:
                        # 没找到可修复的候选，执行跳过逻辑
                        skip_count += 1
                        if skip_count <= max_skip:
                            if skip_count == 1 or z_idx % 10 == 0:
                                print(f"    {direction_name} z_idx={z_idx}: {reason} [跳过 {skip_count}/{max_skip}]")
                            continue
                        else:
                            print(f"    {direction_name}传播终止: 连续{max_skip}次异常")
                            break
                else:
                    # 正常情况，重置跳过计数
                    skip_count = 0
                
                # 找到有效候选，保存结果
                # best_cand 可能是：(label_id, centroid, area) 或 ([label_ids], centroid, area)
                selected_labels[z_idx] = best_cand[:3]
                
                # 改进4&5：更新质心历史和速度向量
                current_centroid = best_cand[1]
                current_area = best_cand[2]
                
                # 更新速度向量（质心移动趋势）
                velocity = (
                    current_centroid[0] - prev_centroid[0],
                    current_centroid[1] - prev_centroid[1]
                )
                
                # 更新历史记录
                centroid_history.append(current_centroid)
                area_history.append(current_area)
                
                # 保持窗口大小
                if len(centroid_history) > window_size * 2:
                    centroid_history = centroid_history[-window_size * 2:]
                    area_history = area_history[-window_size * 2:]
                
                prev_z_idx = z_idx
                prev_centroid = current_centroid
                prev_area = current_area
                
                if z_idx % 10 == 0:  # 每10个切片打印一次
                    label_info = best_cand[0] if not isinstance(best_cand[0], list) else f"{best_cand[0]}"
                    print(f"    {direction_name} z_idx={z_idx}: label={label_info}, "
                          f"质心距离={centroid_dist:.1f}px, 面积比={area_ratio:.2f}")
        
        # 向头侧传播(Z增加)
        head_range = range(start_z_idx + 1, len(self.slices_data))
        propagate_selection(head_range, "头侧")
        
        # 向足侧传播(Z减少)
        foot_range = range(start_z_idx - 1, -1, -1)
        propagate_selection(foot_range, "足侧")
        
        print(f"\n✓ 传播完成,共选择了 {len(selected_labels)} 个切片")
        
        # 5. 检测异常切片
        print("\n4. 异常检测...")
        
        selected_z_indices = sorted(selected_labels.keys())
        
        if len(selected_z_indices) < 3:
            print("  切片数太少,跳过异常检测")
            return selected_labels
        
        # 检查质心连续性
        centroids_list = [selected_labels[z_idx][1] for z_idx in selected_z_indices]
        
        # 计算质心位移
        displacements = []
        for i in range(1, len(centroids_list)):
            dx = centroids_list[i][0] - centroids_list[i-1][0]
            dy = centroids_list[i][1] - centroids_list[i-1][1]
            dist = np.sqrt(dx**2 + dy**2)
            displacements.append(dist)
        
        if displacements:
            mean_disp = np.mean(displacements)
            std_disp = np.std(displacements)
            
            print(f"  质心位移: 均值={mean_disp:.2f}px, 标准差={std_disp:.2f}px")
            
            # 检测异常位移(3倍标准差)
            outliers = []
            for i, disp in enumerate(displacements):
                if disp > mean_disp + 3 * std_disp:
                    z_idx = selected_z_indices[i+1]
                    outliers.append((z_idx, disp))
            
            if outliers:
                print(f"  ⚠ 发现 {len(outliers)} 个异常切片:")
                for z_idx, disp in outliers:
                    print(f"    z_idx={z_idx}, 位移={disp:.2f}px")
            else:
                print("  ✓ 未发现异常切片")
        
        return selected_labels
    
    def _extract_trachea_3d_flood_fill(self, start_z_physical=-100.0, start_z_idx=None):
        """
        使用3D连通性分析提取气管体积 - 像充气一样找到所有连通区域
        
        核心思想：在气管的三维模型中充气，气体能到达的地方都是连通的
        
        策略:
        1. 构建3D二值体数据（所有切片的低密度区域）
        2. 选择种子点（Z≈-100mm处的气管中心）
        3. 使用3D连通域分析（scipy.ndimage.label）
        4. 提取与种子点3D连通的区域
        
        返回:
            selected_labels: {z_idx: (label_id, centroid, area, mask), ...}
        """
        from scipy import ndimage
        
        print("\n" + "="*60)
        print("步骤: 3D连通性分析（充气法）")
        print("="*60)
        
        num_slices = len(self.slices_data)
        if num_slices == 0:
            return {}
        
        # 获取ROI尺寸
        height, width = self.slices_data[0][2].shape
        center_y, center_x = height // 2, width // 2
        half_roi = self.roi_size // 2
        y1 = max(0, center_y - half_roi)
        y2 = min(height, center_y + half_roi)
        x1 = max(0, center_x - half_roi)
        x2 = min(width, center_x + half_roi)
        roi_h, roi_w = y2 - y1, x2 - x1
        
        print(f"1. 构建3D二值体数据...")
        print(f"   切片数: {num_slices}, ROI尺寸: {roi_h}×{roi_w}")
        
        # 构建3D二值体数据
        volume_binary = np.zeros((num_slices, roi_h, roi_w), dtype=np.uint8)
        slice_thresholds = []  # 记录每个切片的阈值
        
        # ⭐ Z屏障：限制充气范围，防止扩散到Z>50mm的区域
        z_barrier = 50.0  # 屏障位置（mm）
        barrier_slices = 0
        
        for z_idx in range(num_slices):
            z_pos, ds, pixel_array = self.slices_data[z_idx]
            
            # ⭐ 屏障检查：Z > barrier 的切片不参与充气
            if z_pos > z_barrier:
                slice_thresholds.append(None)
                barrier_slices += 1
                continue
            
            # 转换为HU值
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                hu_image = pixel_array.astype(np.float32) * slope + intercept
            else:
                hu_image = pixel_array.astype(np.float32)
            
            # 提取ROI
            roi = hu_image[y1:y2, x1:x2]
            
            # 二值化：
            # - 默认：分位阈值（只选择最黑的气管区域）
            # - 可选：固定阈值（窗位窗宽归一化后的[0,1]阈值）
            non_bg = roi[roi > -1900]
            if len(non_bg) < 100:
                slice_thresholds.append(None)
                continue
            
            if self.fixed_threshold is not None:
                # 与 step3_generate_mesh 保持一致的窗位窗宽归一化
                window_center, window_width = -600, 1500
                img_min = window_center - window_width / 2
                img_max = window_center + window_width / 2
                roi_windowed = np.clip(roi, img_min, img_max)
                roi_windowed = (roi_windowed - img_min) / (img_max - img_min)  # [0,1]
                threshold = float(self.fixed_threshold)
                slice_thresholds.append(threshold)
                binary = ((roi > -1900) & (roi_windowed < threshold)).astype(np.uint8)
            else:
                threshold = np.percentile(non_bg, self.percentile)
                slice_thresholds.append(threshold)
                binary = ((roi > -1900) & (roi < threshold)).astype(np.uint8)
            
            # 边缘清零
            binary[0, :] = 0
            binary[-1, :] = 0
            binary[:, 0] = 0
            binary[:, -1] = 0
            
            # 形态学操作（只膨胀，扩散白色区域）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
            binary = cv2.dilate(binary, kernel, iterations=1)
            
            volume_binary[z_idx] = binary
        
        print(f"   二值化完成，非零体素: {np.sum(volume_binary > 0):,}")
        print(f"   ⭐ Z屏障: {z_barrier}mm，屏蔽了{barrier_slices}个切片")
        
        # 1.5 3D形态学闭合操作 - 封闭小孔防止泄漏
        print(f"\n1.5 3D形态学闭合（封闭小孔）...")
        from scipy.ndimage import binary_closing, binary_erosion
        
        # 使用小的3D结构元素进行闭合操作
        structure_close = ndimage.generate_binary_structure(3, 1)  # 6邻域
        volume_closed = binary_closing(volume_binary, structure=structure_close, iterations=self.closing_iters)
        
        # 再轻微腐蚀，移除闭合时可能产生的伪影
        if self.erosion_iters > 0:
            volume_binary = binary_erosion(
                volume_closed, structure=structure_close, iterations=self.erosion_iters
            ).astype(np.uint8)
        else:
            volume_binary = volume_closed.astype(np.uint8)
        
        closed_voxels = np.sum(volume_binary > 0)
        print(f"   闭合后体素: {closed_voxels:,}")
        
        # 2. 选择种子点
        print(f"\n2. 选择种子点...")
        
        if start_z_idx is not None:
            seed_z = start_z_idx
        else:
            # 找最接近start_z_physical的切片
            min_diff = float('inf')
            seed_z = num_slices // 2
            for z_idx in range(num_slices):
                z_pos = self.slices_data[z_idx][0]
                diff = abs(z_pos - start_z_physical)
                if diff < min_diff:
                    min_diff = diff
                    seed_z = z_idx
        
        seed_z_pos = self.slices_data[seed_z][0]
        print(f"   种子切片: z_idx={seed_z}, Z={seed_z_pos:.1f}mm")
        
        # 在种子切片中找到最佳种子点（最靠近中心的连通域）
        seed_slice = volume_binary[seed_z]
        num_labels_2d, labels_2d, stats_2d, centroids_2d = cv2.connectedComponentsWithStats(
            seed_slice, connectivity=8
        )
        
        if num_labels_2d <= 1:
            print("   ✗ 种子切片没有找到候选！")
            return {}
        
        # 找最靠近中心的连通域
        roi_center = (roi_w / 2, roi_h / 2)
        best_seed_label = 1
        min_dist = float('inf')
        
        for lid in range(1, num_labels_2d):
            area = stats_2d[lid, cv2.CC_STAT_AREA]
            if area < 100:  # 过滤小噪声
                continue
            cx, cy = centroids_2d[lid]
            dist = np.sqrt((cx - roi_center[0])**2 + (cy - roi_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                best_seed_label = lid
                seed_x, seed_y = int(cx), int(cy)
        
        print(f"   种子点: (z={seed_z}, y={seed_y}, x={seed_x}), 距中心={min_dist:.1f}px")
        
        # 3. 3D连通域分析
        print(f"\n3. 执行3D连通域分析（26邻域）...")
        
        # 使用26邻域连通（包括对角线）
        structure = ndimage.generate_binary_structure(3, 3)  # 26邻域
        labeled_3d, num_features = ndimage.label(volume_binary, structure=structure)
        
        print(f"   发现 {num_features} 个3D连通域")
        
        # 找到包含种子点的连通域
        seed_label_3d = labeled_3d[seed_z, seed_y, seed_x]
        
        if seed_label_3d == 0:
            # 种子点不在任何连通域中，尝试在附近搜索
            print(f"   ⚠ 种子点不在连通域中，搜索附近...")
            search_radius = 10
            found = False
            for dz in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        nz, ny, nx = seed_z + dz, seed_y + dy, seed_x + dx
                        if 0 <= nz < num_slices and 0 <= ny < roi_h and 0 <= nx < roi_w:
                            if labeled_3d[nz, ny, nx] > 0:
                                seed_label_3d = labeled_3d[nz, ny, nx]
                                seed_z, seed_y, seed_x = nz, ny, nx
                                found = True
                                break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                print("   ✗ 无法找到有效种子点！")
                return {}
            print(f"   找到种子点: (z={seed_z}, y={seed_y}, x={seed_x})")

        # 记录本次3D分析最终使用的种子切片（供网页初始切片/调试使用）
        self.seed_z_idx = int(seed_z)
        self.seed_z_mm = float(self.slices_data[seed_z][0]) if 0 <= seed_z < len(self.slices_data) else None
        
        # 提取气管3D连通域
        trachea_mask_3d = (labeled_3d == seed_label_3d)
        trachea_volume_raw = np.sum(trachea_mask_3d)
        
        print(f"   气管连通域(原始): label={seed_label_3d}, 体积={trachea_volume_raw:,}体素")
        
        # ============================================================
        # ⭐ 泄漏控制策略
        # ============================================================
        print(f"\n4. 应用泄漏控制策略...")
        
        # 注意：所有体素已在ROI范围内（roi_h × roi_w），无需额外边界检查
        
        # 策略1: 面积异常检测 - 逐层检查，面积突然暴增说明泄漏
        print(f"   策略1-面积异常检测...")
        area_history = []
        leaked_slices = 0
        
        # 创建坐标网格（用于面积异常检测时的距离计算）
        yy, xx = np.ogrid[:roi_h, :roi_w]
        
        for z_idx in range(num_slices):
            slice_mask = trachea_mask_3d[z_idx]
            current_area = np.sum(slice_mask)
            
            if current_area < 30:
                continue
            
            # 计算期望面积（基于历史）
            if len(area_history) >= 3:
                avg_area = np.mean(area_history[-5:])  # 最近5层的平均
                max_allowed = avg_area * 3.0  # 最大允许3倍
                
                if current_area > max_allowed and current_area > 2000:
                    # 面积异常，可能泄漏 - 只保留中心区域
                    coords = np.where(slice_mask > 0)
                    if len(coords[0]) > 0:
                        cy, cx = np.mean(coords[0]), np.mean(coords[1])
                        # 创建以质心为中心的圆形掩码
                        expected_radius = np.sqrt(avg_area / np.pi)
                        slice_dist = np.sqrt((yy - cy)**2 + (xx - cx)**2)
                        trachea_mask_3d[z_idx] = slice_mask & (slice_dist <= expected_radius * 1.5)
                        leaked_slices += 1
            
            # 更新面积历史（使用过滤后的面积）
            filtered_area = np.sum(trachea_mask_3d[z_idx])
            if filtered_area > 30:
                area_history.append(filtered_area)
        
        print(f"   检测到{leaked_slices}个疑似泄漏切片并已修正")
        
        trachea_volume = np.sum(trachea_mask_3d)
        print(f"   泄漏控制后体积: {trachea_volume:,}体素 (移除{trachea_volume_raw - trachea_volume:,})")
        
        # 5. 提取每个切片的结果（进行圆形度评分，提取中心线）
        print(f"\n5. 提取各切片气管区域（圆形度评分）...")
        
        selected_labels = {}
        centerline_points = []  # [(z_idx, cy_roi, cx_roi), ...] 用于中心线约束
        valid_slices = 0
        
        # ⭐ 顺序追踪：初始锚点为图像中心，之后随每个成功切片更新
        img_center = (roi_w / 2, roi_h / 2)
        max_dist = np.sqrt(roi_w**2 + roi_h**2) / 2
        prev_center = img_center  # 上一个有效切片的质心（初始为图像中心）
        
        for z_idx in range(num_slices):
            slice_mask = trachea_mask_3d[z_idx].astype(np.uint8)
            
            if np.sum(slice_mask) < 30:  # 面积太小，跳过
                continue
            
            # ⭐ 对mask进行2D连通域分析，然后进行圆形度评分选择最佳区域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                slice_mask, connectivity=8
            )
            
            if num_labels <= 1:  # 没有连通域
                continue
            
            # 对每个连通域计算圆形度评分
            candidates = []
            for label_id in range(1, num_labels):  # 跳过背景(0)
                area = stats[label_id, cv2.CC_STAT_AREA]
                if area < 30:  # 过滤小噪声
                    continue
                
                # 提取该连通域的mask
                region_mask = (labels == label_id).astype(np.uint8)
                
                # 计算圆形度
                contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    perimeter = cv2.arcLength(contours[0], True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                    else:
                        circularity = 0
                else:
                    circularity = 0
                
                # ⭐ 计算距离上一个有效切片质心（顺序追踪锚点）
                cx_roi, cy_roi = centroids[label_id]
                dist = np.sqrt((cx_roi - prev_center[0])**2 + (cy_roi - prev_center[1])**2)
                
                # 综合评分：圆形度权重0.6，距离权重0.4（距离归一化到0-1，越近越好）
                dist_score = 1.0 - (dist / max_dist)  # 归一化，越近分数越高
                score = circularity * 0.6 + dist_score * 0.4
                
                candidates.append((label_id, (cx_roi, cy_roi), area, region_mask, circularity, dist, score))
            
            if not candidates:
                continue
            
            # 选择得分最高的候选（圆形度评分目标）
            best_candidate = max(candidates, key=lambda c: c[6])  # c[6]是score
            
            # 使用选中区域的质心（ROI坐标）
            cx_roi, cy_roi = best_candidate[1]
            
            # ⭐ 更新顺序追踪锚点为当前切片质心
            prev_center = (cx_roi, cy_roi)
            
            # 添加到中心线点列表
            centerline_points.append((z_idx, cy_roi, cx_roi))
            
            # 转换为全局坐标
            global_cx = cx_roi + x1
            global_cy = cy_roi + y1
            
            area = best_candidate[2]
            selected_mask = best_candidate[3]  # 使用选中区域的mask
            
            # 存储结果（使用特殊label_id=-1表示3D连通域）
            selected_labels[z_idx] = (-1, (global_cx, global_cy), area, selected_mask)
            valid_slices += 1
        
        print(f"\n✓ 3D连通性分析完成!")
        print(f"  - 有效切片: {valid_slices}/{num_slices}")
        print(f"  - Z轴范围: {min(selected_labels.keys()) if selected_labels else 0} 到 {max(selected_labels.keys()) if selected_labels else 0}")
        print(f"  - 总体积: {trachea_volume:,}体素")
        
        # 5.5. 中心线平滑和异常点检测，对偏离轨迹的切片重新分析
        if len(centerline_points) >= 5:
            print(f"\n5.5. 中心线平滑和异常点检测...")
            centerline_array = np.array(centerline_points)  # (N, 3): [z_idx, y, x]
            
            # 按z_idx排序
            sorted_indices = np.argsort(centerline_array[:, 0])
            centerline_array = centerline_array[sorted_indices]
            
            # 滑动窗口平滑（窗口大小=5）
            window_size = 5
            half_window = window_size // 2
            centerline_smoothed = centerline_array.copy()
            
            for i in range(len(centerline_array)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(centerline_array), i + half_window + 1)
                window = centerline_array[start_idx:end_idx]
                # 对x和y坐标进行平滑
                centerline_smoothed[i, 1] = np.mean(window[:, 1])  # y
                centerline_smoothed[i, 2] = np.mean(window[:, 2])  # x
            
            # 计算每个点到平滑轨迹的距离
            distances_to_trajectory = []
            for i in range(len(centerline_array)):
                # 计算到平滑轨迹的距离
                if i > 0 and i < len(centerline_array) - 1:
                    # 使用相邻两点插值
                    prev_point = centerline_smoothed[i-1, 1:3]  # [y, x]
                    next_point = centerline_smoothed[i+1, 1:3]
                    trajectory_point = (prev_point + next_point) / 2
                else:
                    trajectory_point = centerline_smoothed[i, 1:3]
                
                current_point = centerline_array[i, 1:3]
                dist = np.linalg.norm(current_point - trajectory_point)
                distances_to_trajectory.append(dist)
            
            distances_to_trajectory = np.array(distances_to_trajectory)
            
            # 检测异常点：距离超过中位数+2倍标准差
            median_dist = np.median(distances_to_trajectory)
            std_dist = np.std(distances_to_trajectory)
            threshold_dist = median_dist + 2 * std_dist
            
            outlier_indices = np.where(distances_to_trajectory > threshold_dist)[0]
            outlier_z_indices = centerline_array[outlier_indices, 0].astype(int)
            
            print(f"   平滑轨迹计算完成")
            print(f"   异常点阈值: {threshold_dist:.1f}px (中位数={median_dist:.1f}, 标准差={std_dist:.1f})")
            print(f"   检测到{len(outlier_indices)}个异常切片: {outlier_z_indices.tolist()}")
            
            # 对异常切片重新进行2D分析（使用更宽松的参数）
            if len(outlier_z_indices) > 0:
                print(f"\n   对异常切片重新分析（降低二值化阈值）...")
                refined_count = 0
                
                for outlier_z_idx in outlier_z_indices:
                    if outlier_z_idx < 0 or outlier_z_idx >= num_slices:
                        continue
                    
                    # 获取该切片的HU值
                    if 0 <= outlier_z_idx < len(self.slices_data):
                        slice_data = self.slices_data[outlier_z_idx]
                        z_pos = slice_data[0]
                        ds = slice_data[1]
                        pixel_array = slice_data[2]
                        
                        # 转换为HU
                        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                            slope = float(ds.RescaleSlope)
                            intercept = float(ds.RescaleIntercept)
                            hu_image = pixel_array.astype(np.float32) * slope + intercept
                        else:
                            hu_image = pixel_array.astype(np.float32)
                        
                        # 提取ROI
                        roi_hu = hu_image[y1:y2, x1:x2]
                        
                        # ⭐ 使用更宽松的二值化阈值（降低percentile，例如从40%降到35%）
                        relaxed_percentile = max(10, self.percentile - 5)  # 降低5个百分点
                        non_bg = roi_hu[roi_hu > -1900]
                        
                        if len(non_bg) < 100:
                            continue
                        
                        threshold = np.percentile(non_bg, relaxed_percentile)
                        binary = ((roi_hu > -1900) & (roi_hu < threshold)).astype(np.uint8)
                        
                        # 边缘清零
                        binary[0, :] = 0
                        binary[-1, :] = 0
                        binary[:, 0] = 0
                        binary[:, -1] = 0
                        
                        # 形态学操作
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
                        binary = cv2.dilate(binary, kernel, iterations=1)
                        
                        # 2D连通域分析
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                            binary, connectivity=8
                        )
                        
                        if num_labels <= 1:
                            continue
                        
                        # 计算每个连通域的圆形度评分
                        candidates = []
                        max_dist = np.sqrt(roi_w**2 + roi_h**2) / 2
                        
                        # ⭐ 顺序追踪：使用该异常切片之前最近的有效切片质心作为锚点
                        prev_valid_mask = centerline_array[:, 0] < outlier_z_idx
                        if np.any(prev_valid_mask):
                            prev_valid_indices = np.where(prev_valid_mask)[0]
                            prev_center_y = centerline_array[prev_valid_indices[-1], 1]
                            prev_center_x = centerline_array[prev_valid_indices[-1], 2]
                        else:
                            prev_center_y, prev_center_x = roi_h / 2, roi_w / 2
                        
                        for label_id in range(1, num_labels):
                            area = stats[label_id, cv2.CC_STAT_AREA]
                            if area < 30:
                                continue
                            
                            region_mask = (labels == label_id).astype(np.uint8)
                            
                            # 计算圆形度
                            contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                perimeter = cv2.arcLength(contours[0], True)
                                if perimeter > 0:
                                    circularity = 4 * np.pi * area / (perimeter ** 2)
                                else:
                                    circularity = 0
                            else:
                                circularity = 0
                            
                            # ⭐ 计算距离前一有效切片质心（顺序追踪锚点）
                            cx_roi, cy_roi = centroids[label_id]
                            dist = np.sqrt((cx_roi - prev_center_x)**2 + (cy_roi - prev_center_y)**2)
                            
                            # 计算到平滑轨迹的距离（作为额外约束）
                            trajectory_y, trajectory_x = centerline_smoothed[np.argmin(np.abs(centerline_array[:, 0] - outlier_z_idx)), 1:3]
                            dist_to_trajectory = np.sqrt((cy_roi - trajectory_y)**2 + (cx_roi - trajectory_x)**2)
                            
                            # 综合评分：圆形度权重0.4，距离前一切片权重0.3，距离轨迹权重0.3
                            dist_score = 1.0 - (dist / max_dist)
                            trajectory_score = 1.0 - min(1.0, dist_to_trajectory / 50.0)  # 50px内得满分
                            score = circularity * 0.4 + dist_score * 0.3 + trajectory_score * 0.3
                            
                            candidates.append((label_id, (cx_roi, cy_roi), area, region_mask, circularity, dist, dist_to_trajectory, score))
                        
                        if candidates:
                            # 选择得分最高的候选
                            best_candidate = max(candidates, key=lambda c: c[7])  # c[7]是score
                            cx_roi, cy_roi = best_candidate[1]
                            
                            # 更新中心线点
                            centerline_idx = np.where(centerline_array[:, 0] == outlier_z_idx)[0]
                            if len(centerline_idx) > 0:
                                centerline_array[centerline_idx[0], 1] = cy_roi
                                centerline_array[centerline_idx[0], 2] = cx_roi
                                
                                # 更新selected_labels
                                global_cx = cx_roi + x1
                                global_cy = cy_roi + y1
                                selected_labels[outlier_z_idx] = (-1, (global_cx, global_cy), best_candidate[2], best_candidate[3])
                                refined_count += 1
                                
                                print(f"     切片{outlier_z_idx}: 重新分析完成，新中心=({cx_roi:.1f}, {cy_roi:.1f}), "
                                      f"圆形度={best_candidate[4]:.3f}, 距轨迹={best_candidate[6]:.1f}px")
                        else:
                            # ⭐ 如果重新分析没有找到合适的区域，保留原始结果
                            if outlier_z_idx in selected_labels:
                                # 保留原始的中心线点和selected_labels
                                print(f"     切片{outlier_z_idx}: 重新分析未找到更好区域，保留原始结果")
                            else:
                                # 如果原始结果也不存在，至少保留中心线点（使用平滑轨迹的预测值）
                                centerline_idx = np.where(centerline_array[:, 0] == outlier_z_idx)[0]
                                if len(centerline_idx) > 0:
                                    # 使用平滑轨迹的预测值作为中心
                                    trajectory_y, trajectory_x = centerline_smoothed[centerline_idx[0], 1:3]
                                    centerline_array[centerline_idx[0], 1] = trajectory_y
                                    centerline_array[centerline_idx[0], 2] = trajectory_x
                                    
                                    # 创建一个空的mask（至少保留这个切片在selected_labels中）
                                    empty_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
                                    global_cx = trajectory_x + x1
                                    global_cy = trajectory_y + y1
                                    selected_labels[outlier_z_idx] = (-1, (global_cx, global_cy), 0, empty_mask)
                                    print(f"     切片{outlier_z_idx}: 重新分析未找到区域，使用轨迹预测值")
                
                print(f"   ✓ 重新分析了{refined_count}个异常切片")
                
                # 更新centerline_points
                centerline_points = [(int(row[0]), row[1], row[2]) for row in centerline_array]
        
        # 6. 应用中心线距离约束（使用平滑后的中心线）
        if len(centerline_points) >= 3:
            print(f"\n6. 应用中心线距离约束...")
            centerline_array = np.array(centerline_points)  # (N, 3): [z_idx, y, x]
            
            # 对中心线进行最终平滑（用于距离约束）
            sorted_indices = np.argsort(centerline_array[:, 0])
            centerline_array = centerline_array[sorted_indices]
            
            # 滑动窗口平滑
            window_size = 5
            half_window = window_size // 2
            centerline_smooth = centerline_array.copy()
            for i in range(len(centerline_array)):
                start_idx = max(0, i - half_window)
                end_idx = min(len(centerline_array), i + half_window + 1)
                window = centerline_array[start_idx:end_idx]
                centerline_smooth[i, 1] = np.mean(window[:, 1])  # y
                centerline_smooth[i, 2] = np.mean(window[:, 2])  # x
            
            # 估计气管半径（基于选中区域的平均面积）
            avg_area = np.mean([selected_labels[z_idx][2] for z_idx in selected_labels.keys()])
            estimated_radius = np.sqrt(avg_area / np.pi)
            max_distance = 50.0  # 固定最大允许距离（像素）
            
            print(f"   中心线点数: {len(centerline_points)} (已平滑)")
            print(f"   平均面积: {avg_area:.0f}px, 估计半径: {estimated_radius:.1f}px")
            print(f"   最大允许距离: {max_distance:.1f}px")
            
            # 对每个体素计算到中心线的距离（使用平滑后的中心线）
            removed_by_centerline = 0
            trachea_mask_filtered = np.zeros_like(trachea_mask_3d)
            
            for z_idx in range(num_slices):
                slice_mask = trachea_mask_3d[z_idx]
                if np.sum(slice_mask) < 30:
                    continue
                
                # 找到该切片对应的平滑中心线点（最近的）
                slice_centerline = centerline_smooth[centerline_smooth[:, 0] == z_idx]
                
                if len(slice_centerline) == 0:
                    # 如果该切片没有中心线点，使用相邻切片的中心线点（插值）
                    z_diffs = np.abs(centerline_smooth[:, 0] - z_idx)
                    nearest_idx = np.argmin(z_diffs)
                    center_y, center_x = centerline_smooth[nearest_idx, 1], centerline_smooth[nearest_idx, 2]
                else:
                    center_y, center_x = slice_centerline[0, 1], slice_centerline[0, 2]
                
                # 计算该切片每个点到中心线的距离（2D距离）
                y_coords, x_coords = np.where(slice_mask > 0)
                if len(y_coords) == 0:
                    continue
                
                distances = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
                
                # 只保留距离在阈值内的点
                valid_mask = distances <= max_distance
                valid_y = y_coords[valid_mask]
                valid_x = x_coords[valid_mask]
                
                # 更新掩码
                for vy, vx in zip(valid_y, valid_x):
                    trachea_mask_filtered[z_idx, vy, vx] = 1
                
                removed_count = np.sum(slice_mask) - len(valid_y)
                removed_by_centerline += removed_count
            
            # 更新掩码
            trachea_mask_3d = trachea_mask_filtered
            print(f"   中心线约束移除: {removed_by_centerline:,}体素")
            
            # 更新体积
            trachea_volume = np.sum(trachea_mask_3d)
            print(f"   中心线约束后体积: {trachea_volume:,}体素")
            
            # ⭐ 更新selected_labels中的mask，使用过滤后的trachea_mask_3d
            print(f"   更新selected_labels中的mask...")
            updated_labels_count = 0
            for z_idx in list(selected_labels.keys()):
                if z_idx < num_slices:
                    filtered_slice_mask = trachea_mask_3d[z_idx].astype(np.uint8)
                    filtered_area = np.sum(filtered_slice_mask)
                    
                    if filtered_area >= 30:  # 只更新有有效区域的切片
                        # 更新selected_labels中的mask和面积
                        old_label_data = selected_labels[z_idx]
                        selected_labels[z_idx] = (
                            old_label_data[0],  # label_id
                            old_label_data[1],  # centroid (保持不变)
                            filtered_area,      # 更新面积
                            filtered_slice_mask  # 更新mask
                        )
                        updated_labels_count += 1
                    elif filtered_area < 30:
                        # 如果过滤后面积太小，保留但标记为小面积
                        old_label_data = selected_labels[z_idx]
                        selected_labels[z_idx] = (
                            old_label_data[0],
                            old_label_data[1],
                            filtered_area,  # 即使很小也保留
                            filtered_slice_mask
                        )
                        updated_labels_count += 1
            
            print(f"   已更新{updated_labels_count}个切片的mask和面积")
            
            # ⭐ 提取中心线到3D可视化坐标系（使用平滑后的中心线）
            if self.downsample_size is not None and self.original_size is not None:
                scale_factor = self.downsample_size / self.original_size[0]
                centerline_world = []
                for row in centerline_smooth:
                    z_idx = int(row[0])
                    cy_roi, cx_roi = row[1], row[2]
                    if 0 <= z_idx < len(self.slices_data):
                        z_mm = self.slices_data[z_idx][0]
                    else:
                        z_mm = self.slices_data[-1][0] if z_idx >= len(self.slices_data) else self.slices_data[0][0]
                    x_ds = (cx_roi + x1) * scale_factor
                    y_ds = (cy_roi + y1) * scale_factor
                    centerline_world.append((x_ds, y_ds, z_mm))
                self.centerline_world = np.array(centerline_world, dtype=np.float32)
                print(f"   中心线已提取到3D坐标系: {len(centerline_world)}点 (平滑后)")
                
                # ⭐ 保存ROI坐标系中的中心线数据（用于2D分析时的距离筛选）
                self.centerline_roi = centerline_smooth.copy()  # (N, 3): [z_idx, y_roi, x_roi]
                print(f"   中心线ROI数据已保存: {len(centerline_smooth)}点")
            else:
                self.centerline_world = None
                self.centerline_roi = None
        else:
            print(f"   ⚠ 中心线点数太少({len(centerline_points)})，跳过中心线约束")
            self.centerline_world = None
            self.centerline_roi = None
        
        # 保存3D掩码供后续使用
        self.trachea_mask_3d = trachea_mask_3d
        self.roi_offset = (x1, y1)
        
        return selected_labels
    
    def _add_flood_fill_mesh_to_figure(self, fig, *, color='red', opacity=0.3, name='充气法区域', visible=True):
        """
        在3D图中添加充气法选中区域的可视化
        用半透明红色网格显示，帮助调试充气是否泄漏
        """
        if self.trachea_mask_3d is None:
            return
        
        print(f"\n生成充气法3D可视化...")
        
        # 使用Marching Cubes从3D掩码生成网格
        try:
            # 对掩码进行高斯平滑使网格更光滑
            from scipy.ndimage import gaussian_filter
            smoothed_mask = gaussian_filter(self.trachea_mask_3d.astype(np.float32), sigma=0.5)
            
            verts, faces, normals, values = measure.marching_cubes(
                smoothed_mask,
                level=0.5,
                step_size=1,
                allow_degenerate=False
            )
            
            print(f"  充气区域网格: {len(verts):,}顶点, {len(faces):,}三角形")
            
            # ⭐ 坐标转换：ROI局部坐标 → 原始全局坐标 → 降采样坐标
            x1_orig, y1_orig = self.roi_offset  # 原始分辨率下的ROI偏移
            
            # Marching Cubes输出: verts[:, 0]=Z_idx, verts[:, 1]=Y_roi, verts[:, 2]=X_roi
            # 需要转换为降采样后的坐标系（与主网格对齐）
            
            # 计算缩放比例：原始分辨率 → 降采样分辨率
            scale_factor = self.downsample_size / self.original_size[0]  # 例如 256/512 = 0.5
            
            print(f"  坐标转换: 原始分辨率{self.original_size} → 降采样{self.downsample_size}")
            print(f"  缩放因子: {scale_factor:.3f}")
            print(f"  ROI偏移(原始): x1={x1_orig}, y1={y1_orig}")
            
            new_verts = np.zeros_like(verts)
            
            # X坐标: (ROI_X + 原始偏移) * 缩放 = 降采样X
            new_verts[:, 0] = (verts[:, 2] + x1_orig) * scale_factor
            
            # Y坐标: (ROI_Y + 原始偏移) * 缩放 = 降采样Y  
            new_verts[:, 1] = (verts[:, 1] + y1_orig) * scale_factor
            
            # Z坐标转换为物理坐标(mm)
            z_indices = verts[:, 0].astype(int)
            z_physical = np.zeros(len(verts))
            for i, z_idx in enumerate(z_indices):
                if 0 <= z_idx < len(self.slices_data):
                    z_physical[i] = self.slices_data[z_idx][0]
                else:
                    z_physical[i] = self.slices_data[-1][0] if z_idx >= len(self.slices_data) else self.slices_data[0][0]
            
            new_verts[:, 2] = z_physical

            self.trachea_lumen_verts = new_verts.astype(np.float64, copy=True)
            self.trachea_lumen_faces = faces.astype(np.int64, copy=True)
            
            # 添加到图中
            fig.add_trace(go.Mesh3d(
                x=new_verts[:, 0],
                y=new_verts[:, 1],
                z=new_verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=opacity,
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.5,
                    roughness=0.9,
                    specular=0.1
                ),
                name=name,
                showlegend=True,
                visible=visible
            ))
            # 反向绕序的同一等值面：法线指向气腔内侧，便于“相机在腔内”时仍能看到壁（WebGL 正面）
            faces_in = faces[:, [0, 2, 1]].copy()
            fig.add_trace(go.Mesh3d(
                x=new_verts[:, 0],
                y=new_verts[:, 1],
                z=new_verts[:, 2],
                i=faces_in[:, 0],
                j=faces_in[:, 1],
                k=faces_in[:, 2],
                color='#86efac',
                opacity=0.48,
                lighting=dict(
                    ambient=0.92,
                    diffuse=0.42,
                    roughness=0.92,
                    specular=0.08
                ),
                name='气管内壁(腔内)',
                showlegend=True,
                visible=False,
            ))
            
            print(f"  ✓ 充气法3D网格已添加 ({color}, opacity={opacity})")
            print(f"    范围: X[{new_verts[:,0].min():.1f},{new_verts[:,0].max():.1f}], "
                  f"Y[{new_verts[:,1].min():.1f},{new_verts[:,1].max():.1f}], "
                  f"Z[{new_verts[:,2].min():.1f},{new_verts[:,2].max():.1f}]mm")
            
            # 统计体积信息
            total_voxels = np.sum(self.trachea_mask_3d)
            print(f"    总体积: {total_voxels:,} 体素")
            
            # ⭐ 在3D图上添加中心线（如果已计算）
            if hasattr(self, 'centerline_world') and self.centerline_world is not None and len(self.centerline_world) > 1:
                cl = self.centerline_world
                fig.add_trace(go.Scatter3d(
                    x=cl[:, 0],
                    y=cl[:, 1],
                    z=cl[:, 2],
                    mode='lines+markers',
                    line=dict(color='yellow', width=5),
                    marker=dict(size=3, color='yellow'),
                    name='充气中心线',
                    showlegend=True
                ))
                print(f"  ✓ 充气中心线已添加到3D图 (黄色)")
            
        except Exception as e:
            print(f"  ✗ 生成充气法网格失败: {e}")
            self.trachea_lumen_verts = None
            self.trachea_lumen_faces = None
    
    def step3_generate_mesh(self, downsample_size=256, iso_value=0.5, step_size=2):
        """步骤3: Marching Cubes生成3D网格"""
        print("\n" + "="*60)
        print("步骤 3/4: Marching Cubes生成3D网格")
        print("="*60)
        
        # 创建体数据
        num_slices = len(self.slices_data)
        height = self.slices_data[0][2].shape[0]
        width = self.slices_data[0][2].shape[1]
        
        # 记录尺寸信息
        self.original_size = (height, width)
        self.downsample_size = downsample_size
        
        print(f"原始尺寸: {num_slices}×{height}×{width}")
        print(f"降采样到: {num_slices}×{downsample_size}×{downsample_size}")
        print(f"缩放比例: {downsample_size/height:.3f}")
        
        # 构建体数据(降采样)
        volume = np.zeros((num_slices, downsample_size, downsample_size), dtype=np.float32)
        
        for idx, (z_pos, ds, pixel_array) in enumerate(self.slices_data):
            # 转换为HU值
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                slope = float(ds.RescaleSlope)
                intercept = float(ds.RescaleIntercept)
                hu_image = pixel_array.astype(np.float32) * slope + intercept
            else:
                hu_image = pixel_array.astype(np.float32)
            
            # 窗位窗宽调整
            window_center, window_width = -600, 1500
            img_min = window_center - window_width / 2
            img_max = window_center + window_width / 2
            windowed = np.clip(hu_image, img_min, img_max)
            windowed = (windowed - img_min) / (img_max - img_min)
            
            # 降采样
            from PIL import Image
            img = Image.fromarray((windowed * 255).astype(np.uint8))
            if img.size != (downsample_size, downsample_size):
                img = img.resize((downsample_size, downsample_size), Image.Resampling.LANCZOS)
            volume[idx] = np.array(img) / 255.0
        
        self.volume = volume
        print(f"✓ 体数据已生成: {volume.shape}")
        print(f"  数值范围: [{volume.min():.3f}, {volume.max():.3f}]")
        print(f"  第一个切片shape: {volume[0].shape}")
        print(f"  原始第一个切片shape: {self.slices_data[0][2].shape}")
        
        # 验证坐标对应关系
        print(f"\n🔍 验证volume坐标系统:")
        z_pos, ds, pixel_array = self.slices_data[0]
        slope = float(ds.RescaleSlope) if hasattr(ds, 'RescaleSlope') else 1.0
        intercept = float(ds.RescaleIntercept) if hasattr(ds, 'RescaleIntercept') else 0.0
        hu_image = pixel_array.astype(np.float32) * slope + intercept
        # 测试点：原始(X=168, Y=182) -> 缩放(X=42, Y=45.5≈46)
        orig_x, orig_y = 168, 182
        scale_x, scale_y = int(orig_x * 0.25), int(orig_y * 0.25)
        print(f"  测试点: 原始[Y={orig_y}, X={orig_x}] -> Volume[Y={scale_y}, X={scale_x}]")
        print(f"  原始HU值: {hu_image[orig_y, orig_x]:.1f}")
        print(f"  Volume值: {volume[0, scale_y, scale_x]:.3f}")
        # 测试四个角
        corners = [(0, 0), (0, 127), (127, 0), (127, 127)]
        print(f"  Volume四角值:")
        for y, x in corners:
            print(f"    [Y={y}, X={x}]: {volume[0, y, x]:.3f}")
        
        # Marching Cubes
        print(f"\nMarching Cubes参数:")
        print(f"  等值面: {iso_value}")
        print(f"  步长: {step_size}")
        
        try:
            verts, faces, normals, values = measure.marching_cubes(
                volume,
                level=iso_value,
                step_size=step_size,
                allow_degenerate=False
            )
            
            self.vertices = verts
            self.faces = faces
            
            print(f"\n✓ 网格生成成功:")
            print(f"  顶点数: {len(verts):,}")
            print(f"  三角形数: {len(faces):,}")
            
            return len(verts), len(faces)
            
        except Exception as e:
            print(f"✗ Marching Cubes失败: {e}")
            return 0, 0

    # 图层显隐控制：使用 Plotly 自带 legend（单击隐藏/显示，双击独显）
    
    def step4_create_visualization(self, output_html=None,
                                   show_endoscopy=False,
                                   show_cross_sections=True,
                                   cross_section_interval=10,
                                   use_3d_analysis=False,
                                   use_flood_fill=True,
                                   start_z=-100.0,
                                   start_idx=None,
                                   navigation_line=False,
                                   nav_min_turn_radius_mm=12.0):
        """步骤4: 创建完整的3D可视化(网格+充气中心线/导航线+横截面+虚拟内窥镜)
        
        参数:
            use_3d_analysis: 是否使用3D分析
            use_flood_fill: True=充气法(3D连通性), False=传统传播法
        """
        print("\n" + "="*60)
        print("步骤 4/4: 生成交互式3D可视化")
        print("="*60)
        
        if output_html is None:
            output_html = f"{self.output_name}_3d.html"
        
        # 创建图形
        fig = go.Figure()
        
        # 若启用3D分析，先提取气管体积（供网格/轮廓分析/中心线使用）
        selected_3d_labels = {}
        if use_3d_analysis:
            if use_flood_fill:
                selected_3d_labels = self._extract_trachea_3d_flood_fill(
                    start_z_physical=start_z,
                    start_z_idx=start_idx
                )
                print(f"\n使用3D充气法分析结果 (共{len(selected_3d_labels)}个切片)")
            else:
                selected_3d_labels = self._extract_trachea_3d_volume(
                    start_z_physical=start_z,
                    start_z_idx=start_idx
                )
                print(f"\n使用传统传播法分析结果 (共{len(selected_3d_labels)}个切片)")

        # 1) 其他结构背景（半透明）：强度等值面网格
        #    目的：在3D画面中提供“解剖/其他结构”的空间参照，气管网格在其上叠加显示。
        has_main_mesh = (self.vertices is not None and self.faces is not None)
        if has_main_mesh:
            verts = self.vertices.copy()
            
            # 重新排列坐标轴: 原来的(Z,Y,X) -> 新的(X,Y,Z)
            # Marching Cubes输出: verts[:, 0]=Z切片索引, verts[:, 1]=Y行, verts[:, 2]=X列
            # 我们要: X=原X列, Y=原Y行, Z=实际物理Z坐标(mm)
            new_verts = np.zeros_like(verts)
            new_verts[:, 0] = verts[:, 2]  # X = 原X列
            new_verts[:, 1] = verts[:, 1]  # Y = 原Y行
            
            # 将Z切片索引转换为实际物理Z坐标(mm)
            # verts[:, 0]是切片索引，需要映射到对应的物理Z坐标
            z_indices = verts[:, 0].astype(int)
            z_physical = np.zeros(len(z_indices))
            for i, z_idx in enumerate(z_indices):
                if 0 <= z_idx < len(self.slices_data):
                    z_physical[i] = self.slices_data[z_idx][0]  # 获取物理Z坐标(mm)
                else:
                    # 边界情况：插值
                    z_physical[i] = self.slices_data[-1][0]
            
            new_verts[:, 2] = z_physical  # Z = 实际物理坐标(mm)
            verts = new_verts

            # 半透明背景网格（单色，避免过大的颜色映射开销）
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=self.faces[:, 0],
                j=self.faces[:, 1],
                k=self.faces[:, 2],
                color='#9CA3AF',  # gray-400
                opacity=0.18,     # 其他结构半透明背景
                showscale=False,
                lighting=dict(
                    ambient=0.9,
                    diffuse=0.4,
                    roughness=0.9,
                    specular=0.05
                ),
                name='其他结构(半透明)',
                showlegend=True,
                flatshading=False,
                visible=True
            ))
            print(f"✓ 添加背景网格(其他结构): {len(verts):,}个顶点")
            print(f"  网格范围: X[{verts[:,0].min():.1f},{verts[:,0].max():.1f}], "
                  f"Y[{verts[:,1].min():.1f},{verts[:,1].max():.1f}], "
                  f"Z[{verts[:,2].min():.1f},{verts[:,2].max():.1f}]mm")

        # 2) 气管网格（3D分析开启时优先显示掩码网格）
        used_mask_mesh = False
        if use_3d_analysis and hasattr(self, 'trachea_mask_3d') and self.trachea_mask_3d is not None:
            self._add_flood_fill_mesh_to_figure(
                fig,
                color='green',
                opacity=0.65,
                name='气管区域(3D充气法)',
                visible=True
            )
            used_mask_mesh = True

        # 2.5) 导航线（仅用3D掩码 + 几何代价；插管方向：zmax -> zmin）
        if navigation_line and use_3d_analysis and hasattr(self, 'trachea_mask_3d') and self.trachea_mask_3d is not None:
            ok = self._compute_navigation_path_from_mask(
                min_turn_radius_mm=float(nav_min_turn_radius_mm),
                resample_step_mm=1.0,
                wall_bias_power=1.5,
            )
            if ok and self.navigation_path_plotly is not None and len(self.navigation_path_plotly) > 1:
                nav = self.navigation_path_plotly
                fig.add_trace(go.Scatter3d(
                    x=nav[:, 0],
                    y=nav[:, 1],
                    z=nav[:, 2],
                    mode='lines+markers',
                    line=dict(color='#00D1B2', width=8),
                    marker=dict(
                        size=4,
                        color='#00D1B2',
                        line=dict(width=0.5, color='rgba(0,0,0,0.35)'),
                    ),
                    opacity=1.0,
                    name='导航线(插管路径)',
                    showlegend=True,
                ))
                # 漫游相机 eye 位置（由注入脚本 Plotly.restyle 每帧更新）
                fig.add_trace(go.Scatter3d(
                    x=[float(nav[0, 0])],
                    y=[float(nav[0, 1])],
                    z=[float(nav[0, 2])],
                    mode='markers',
                    marker=dict(
                        size=11,
                        color='#F472B6',
                        symbol='diamond',
                        line=dict(width=2, color='rgba(0,0,0,0.45)'),
                    ),
                    name='相机位置(漫游)',
                    showlegend=True,
                    hovertemplate='相机位置(漫游)<br>X=%{x:.1f}<br>Y=%{y:.1f}<br>Z=%{z:.1f} mm<extra></extra>',
                ))
                n0 = len(nav)
                j0 = min(3, n0 - 1)
                p0 = nav[0].astype(np.float64)
                p1 = nav[j0].astype(np.float64)
                mid0 = p0 + 0.35 * (p1 - p0)
                fig.add_trace(go.Scatter3d(
                    x=[float(p0[0]), float(p1[0])],
                    y=[float(p0[1]), float(p1[1])],
                    z=[float(p0[2]), float(p1[2])],
                    mode='lines',
                    line=dict(color='#FBBF24', width=5),
                    name='视线(漫游)',
                    showlegend=True,
                    hoverinfo='skip',
                ))
                fig.add_trace(go.Scatter3d(
                    x=[float(mid0[0])],
                    y=[float(mid0[1])],
                    z=[float(mid0[2])],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='#FB923C',
                        symbol='cross',
                        line=dict(width=1, color='rgba(0,0,0,0.35)'),
                    ),
                    name='注视点(漫游)',
                    showlegend=True,
                    hovertemplate='注视点(漫游)<br>X=%{x:.1f}<br>Y=%{y:.1f}<br>Z=%{z:.1f} mm<extra></extra>',
                ))
                print(f"✓ 导航线已添加到3D图: {len(nav)}点")
            else:
                print("⚠ 导航线计算失败或点数不足，已跳过绘制")
        # 充气法网格内已绘制「充气中心线」；此处不再重复添加「中心线」图层
        
        # 3. 添加横截面切片 (对所有切片生成分析)
        if show_cross_sections and self.volume is not None:
            num_slices = self.volume.shape[0]
            
            print(f"\n生成所有切片的横截面分析 (共{num_slices}个切片)...")
            
            # 按间隔抽样生成分析（避免把全部切片都塞进HTML导致体积/内存暴涨）
            interval = int(cross_section_interval) if cross_section_interval and cross_section_interval > 0 else 1
            slice_positions = list(range(0, num_slices, interval))
            if (num_slices - 1) not in slice_positions:
                slice_positions.append(num_slices - 1)

            # 确保种子切片一定会被生成（否则“初始选中切片=种子切片”无法满足）
            seed_z_idx = getattr(self, "seed_z_idx", None)
            if seed_z_idx is not None and 0 <= int(seed_z_idx) < num_slices and int(seed_z_idx) not in slice_positions:
                slice_positions.append(int(seed_z_idx))
                slice_positions.sort()
            
            if interval == 1:
                print(f"  将分析所有切片: Z=0 到 Z={num_slices-1}")
            else:
                print(f"  将按间隔分析切片: interval={interval} (共{len(slice_positions)}张)")
            
            added_count = 0
            self.cross_section_analysis = []  # 清空之前的分析
            
            # 用于存储所有切片的轮廓和中心点数据（用于滑块控制）
            all_slice_contours = []
            
            # 滑动窗口面积历史（用于自适应阈值）
            area_history_viz = []
            window_size_viz = 10
            
            for z_idx in slice_positions:
                if z_idx >= len(self.slices_data):
                    continue
                
                # 每10个切片显示一次进度
                if z_idx % 10 == 0:
                    print(f"  处理中... Z={z_idx}/{num_slices-1}", end='\r')
                
                # 获取基本信息
                z_physical = self.slices_data[z_idx][0]  # 物理Z坐标(mm)
                _, ds, pixel_array = self.slices_data[z_idx]
                
                # 转换为HU值
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    slope = float(ds.RescaleSlope)
                    intercept = float(ds.RescaleIntercept)
                    slice_hu = pixel_array.astype(np.float32) * slope + intercept
                else:
                    slice_hu = pixel_array.astype(np.float32)
                
                # 计算ROI坐标（统一使用）
                height, width = slice_hu.shape
                roi_size = self.roi_size
                y1 = max(0, height // 2 - roi_size // 2)
                y2 = min(height, y1 + roi_size)
                x1 = max(0, width // 2 - roi_size // 2)
                x2 = min(width, x1 + roi_size)
                
                # 初始化analysis对象
                should_generate_detailed_images = (z_physical <= 100)
                if z_idx < self.volume.shape[0]:
                    slice_data_ds = self.volume[z_idx, :, :]
                else:
                    slice_data_ds = None
                
                # ============================================================
                # ⭐⭐⭐ 最高优先级：3D充气法指导筛选 ⭐⭐⭐
                # ============================================================
                if use_3d_analysis and z_idx in selected_3d_labels and len(selected_3d_labels[z_idx]) > 3:
                    # 获取3D充气法预计算的mask（原始空间ROI，300×300）
                    mask_3d_original = selected_3d_labels[z_idx][3].astype(np.uint8)
                    total_area_3d = selected_3d_labels[z_idx][2]
                    centroid_3d = selected_3d_labels[z_idx][1]
                    
                    # ⭐ 关键：将mask_3d从原始空间缩放到降采样空间
                    # 原始ROI: 300×300, 降采样ROI: 150×150
                    scale_factor = self.downsample_size / self.original_size[0]  # 0.5
                    roi_size_ds = int(self.roi_size * scale_factor)  # 150
                    
                    # 使用cv2.resize缩放mask
                    mask_3d = cv2.resize(mask_3d_original, (roi_size_ds, roi_size_ds), 
                                        interpolation=cv2.INTER_NEAREST)
                    
                    # ============================================================
                    # 🔗 展示3D连通性如何把堵塞物分裂的区域重新连接
                    # ============================================================
                    
                    # 步骤1: 对缩放后的mask进行2D连通域分析
                    num_labels_in_mask, labels_in_mask, stats_in_mask, centroids_in_mask = cv2.connectedComponentsWithStats(
                        mask_3d, connectivity=8
                    )
                    
                    # 收集所有分离区域的信息
                    separated_regions = []
                    for lid in range(1, num_labels_in_mask):
                        area = stats_in_mask[lid, cv2.CC_STAT_AREA]
                        if area >= 10:  # 缩放后面积变小，降低阈值
                            cx, cy = centroids_in_mask[lid]
                            separated_regions.append({
                                'id': lid,
                                'area': area,
                                'center': (cx, cy)
                            })
                    
                    num_separated = len(separated_regions)
                    
                    # 生成详细分析图像（包含3D连通性信息）
                    if should_generate_detailed_images:
                        analysis = self._generate_cross_section_analysis(
                            z_idx, slice_data_ds, slice_hu, 
                            mask_3d=mask_3d, 
                            separated_regions=separated_regions
                        )
                    else:
                        analysis = {'z_index': z_idx, 'images': {}, 'stats': None}
                    
                    # 展示3D连通性的指导作用（同切面内合并）
                    if z_idx % 20 == 0 or num_separated > 1:
                        print(f"\n    ═══════════════════════════════════════════════════")
                        print(f"    📊 [同切面3D合并] 切片 z_idx={z_idx} (Z={z_physical:.1f}mm)")
                        print(f"    ═══════════════════════════════════════════════════")
                        
                        if num_separated > 1:
                            # 🔗 关键：同一切面内，多个2D分离的域因3D连通而合并
                            print(f"    ⚠️  本切面检测到堵塞物导致的区域分裂!")
                            print(f"    ┌─ 本层2D分析: 发现 {num_separated} 个分离的域")
                            for i, region in enumerate(separated_regions):
                                print(f"    │   域{i+1}: 面积={region['area']}px, 中心=({region['center'][0]:.0f}, {region['center'][1]:.0f})")
                            print(f"    │")
                            print(f"    │  [3D连通性判断]")
                            print(f"    │   → 这{num_separated}个域虽然在本层2D上分离")
                            print(f"    │   → 但通过上下层，它们在3D空间中是连通的")
                            print(f"    │")
                            print(f"    └─ 本层结论: {num_separated}个分离域 → 合并为1个气管区域")
                            print(f"       合并后面积: {total_area_3d}px")
                        else:
                            print(f"    ✓ 本切面气管完整，无分裂 (面积={total_area_3d}px)")
                    
                    # 使用3D充气法的mask作为最终结果
                    mask = mask_3d
                    total_area = total_area_3d
                    
                    # 更新面积历史
                    area_history_viz.append(total_area)
                    if len(area_history_viz) > window_size_viz * 2:
                        area_history_viz = area_history_viz[-window_size_viz * 2:]
                
                else:
                    # ============================================================
                    # 传统方法：2D分析 + 面积筛选（仅当没有3D充气法结果时使用）
                    # ============================================================
                    
                    # 计算降采样ROI尺寸
                    scale_factor = self.downsample_size / self.original_size[0]
                    roi_size_ds = int(self.roi_size * scale_factor)
                    
                    # 生成详细分析图像（无3D信息）
                    if should_generate_detailed_images:
                        analysis = self._generate_cross_section_analysis(z_idx, slice_data_ds, slice_hu)
                    else:
                        analysis = {'z_index': z_idx, 'images': {}, 'stats': None}
                    
                    slice_hu_roi = slice_hu[y1:y2, x1:x2]
                    
                    # 在ROI内计算分位阈值（只选择最黑的气管区域）
                    non_bg = slice_hu_roi[(slice_hu_roi > -500) & (slice_hu_roi < 200)]
                    if len(non_bg) < 100:
                        # ⭐ 即使没有找到合适的区域，也创建一个空的analysis，确保切片被包含
                        mask = np.zeros((roi_size_ds, roi_size_ds), dtype=np.uint8)
                        total_area = 0
                        if analysis and 'stats' in analysis:
                            analysis['stats'] = {
                                'area': 0,
                                'center': (0, 0),
                                'center_roi': (0, 0),
                                'contour_roi': [],
                                'roi_offset': (0, 0)
                            }
                    else:
                        threshold = np.percentile(non_bg, self.percentile)
                        binary_roi = ((slice_hu_roi > -500) & (slice_hu_roi < threshold)).astype(np.uint8)
                        
                        # 调试信息
                        if z_idx == 30:
                            print(f"\n  🔍 横截面Z={z_idx}调试:")
                            print(f"    ROI HU范围: [{slice_hu_roi.min():.1f}, {slice_hu_roi.max():.1f}]")
                            print(f"    40%阈值: {threshold:.1f}")
                            print(f"    二值化后非零点: {np.count_nonzero(binary_roi)}")
                        
                        # 边缘清零
                        binary_roi[0, :] = 0
                        binary_roi[-1, :] = 0
                        binary_roi[:, 0] = 0
                        binary_roi[:, -1] = 0
                        
                        # 形态学操作（只膨胀，扩散白色区域）
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.kernel_size)
                        binary_roi = cv2.dilate(binary_roi, kernel, iterations=1)
                        
                        # 连通域分析
                        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                            binary_roi, connectivity=8
                        )
                        
                        if num_labels <= 1:
                            # ⭐ 即使没有连通域，也创建一个空的mask，确保切片被包含
                            mask = np.zeros((roi_size_ds, roi_size_ds), dtype=np.uint8)
                            total_area = 0
                            if analysis and 'stats' in analysis:
                                analysis['stats'] = {
                                    'area': 0,
                                    'center': (0, 0),
                                    'center_roi': (0, 0),
                                    'contour_roi': [],
                                    'roi_offset': (0, 0)
                                }
                        else:
                            # 面积筛选：下限过滤噪声，上限过滤体外大空气区
                            area_min = 30
                            area_max = 200  # ⭐ >200px 很可能是体外空气，排除
                            
                            # 找最佳候选 - 使用中心线投影作为锚点
                            # ⭐ 获取当前切片对应的中心线投影点（ROI坐标）
                            centerline_anchor_x = None
                            centerline_anchor_y = None
                            
                            if hasattr(self, 'centerline_roi') and self.centerline_roi is not None and len(self.centerline_roi) > 0:
                                # ⭐ 顺序追踪：使用严格在当前切片之前的最近有效切片质心
                                centerline_roi_array = self.centerline_roi
                                prev_cl_mask = centerline_roi_array[:, 0] < z_idx
                                if np.any(prev_cl_mask):
                                    prev_cl_indices = np.where(prev_cl_mask)[0]
                                    nearest_prev_cl_idx = prev_cl_indices[-1]
                                    centerline_anchor_y_roi = centerline_roi_array[nearest_prev_cl_idx, 1]
                                    centerline_anchor_x_roi = centerline_roi_array[nearest_prev_cl_idx, 2]
                                    centerline_anchor_y = centerline_anchor_y_roi + y1
                                    centerline_anchor_x = centerline_anchor_x_roi + x1
                                else:
                                    # 没有前一个切片，使用最近的（可能是后面的切片）
                                    z_diffs = np.abs(centerline_roi_array[:, 0] - z_idx)
                                    nearest_idx = np.argmin(z_diffs)
                                    centerline_anchor_y_roi = centerline_roi_array[nearest_idx, 1]
                                    centerline_anchor_x_roi = centerline_roi_array[nearest_idx, 2]
                                    centerline_anchor_y = centerline_anchor_y_roi + y1
                                    centerline_anchor_x = centerline_anchor_x_roi + x1
                            
                            # 如果没有中心线数据，回退到图像中心
                            if centerline_anchor_x is None or centerline_anchor_y is None:
                                centerline_anchor_y = self.original_size[0] / 2
                                centerline_anchor_x = self.original_size[1] / 2
                            
                            candidates = []
                            for label_id in range(1, num_labels):
                                area = stats[label_id, cv2.CC_STAT_AREA]
                                if area < area_min or area > area_max:  # ⭐ 同时过滤噪声和体外大空气
                                    continue
                                cx, cy = centroids[label_id]
                                global_cy = cy + y1
                                global_cx = cx + x1
                                # ⭐ 使用中心线投影作为锚点计算距离
                                dist = np.sqrt((global_cx - centerline_anchor_x)**2 + (global_cy - centerline_anchor_y)**2)
                                candidates.append((label_id, dist))
                            
                            if not candidates:
                                # ⭐ 即使没有候选，也创建一个空的mask，确保切片被包含
                                mask = np.zeros((roi_size_ds, roi_size_ds), dtype=np.uint8)
                                total_area = 0
                                if analysis and 'stats' in analysis:
                                    analysis['stats'] = {
                                        'area': 0,
                                        'center': (0, 0),
                                        'center_roi': (0, 0),
                                        'contour_roi': [],
                                        'roi_offset': (0, 0)
                                    }
                            else:
                                # 选择候选
                                if len(candidates) == 1:
                                    best_idx = candidates[0][0]
                                else:
                                    best_candidate = min(candidates, key=lambda x: x[1])
                                    best_idx = best_candidate[0]
                                    if z_idx == 30:
                                        print(f"    📊 [距离中心线优先] 多个候选({len(candidates)}个):")
                                        for lid, dist in sorted(candidates, key=lambda x: x[1]):
                                            marker = "✓" if lid == best_idx else " "
                                            print(f"      {marker} ID={lid}: 距离中心线={dist:.1f}px")
                                
                                # 提取mask
                                mask = (labels == best_idx).astype(np.uint8)
                                total_area = stats[best_idx, cv2.CC_STAT_AREA]
                                
                                # 更新面积历史
                                area_history_viz.append(total_area)
                                if len(area_history_viz) > window_size_viz * 2:
                                    area_history_viz = area_history_viz[-window_size_viz * 2:]
                
                # ============================================================
                # 统一的轮廓提取和后续处理
                # ============================================================
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # ⭐ 即使没有轮廓，也继续处理（确保切片被包含）
                if not contours:
                    # 创建空的轮廓数据，但继续处理
                    all_points_roi = []
                    all_points_global = []
                    total_contour_area = 0
                    largest_contour = None
                    points = np.array([]).reshape(0, 2)
                    points_global = np.array([]).reshape(0, 2)
                    
                    # 使用mask的质心作为中心点
                    M = cv2.moments(mask)
                    if M['m00'] != 0:
                        cx_roi = M['m10'] / M['m00']
                        cy_roi = M['m01'] / M['m00']
                        cx_global = cx_roi + x1
                        cy_global = cy_roi + y1
                    else:
                        cx_global = x1 + mask.shape[1] // 2
                        cy_global = y1 + mask.shape[0] // 2
                        cx_roi = cx_global - x1
                        cy_roi = cy_global - y1
                    
                    # 设置空的analysis stats
                    if analysis.get('stats') is None:
                        analysis['stats'] = {
                            'contour_roi': [],
                            'all_contours_roi': [],
                            'roi_offset': (x1, y1),
                            'center': (cx_global, cy_global),
                            'center_roi': (cx_roi, cy_roi),
                            'area': total_area,
                            'num_contours': 0
                        }
                else:
                    # 处理所有轮廓（3D连通的多个区域）
                    all_points_roi = []
                    all_points_global = []
                    total_contour_area = 0
                    
                    for contour in contours:
                        contour_area = cv2.contourArea(contour)
                        if contour_area < 30:  # 过滤极小噪声
                            continue
                        total_contour_area += contour_area
                        pts = contour[:, 0, :]
                        all_points_roi.append(pts)
                        pts_global = pts.copy()
                        pts_global[:, 0] += x1
                        pts_global[:, 1] += y1
                        all_points_global.append(pts_global)
                    
                    if not all_points_roi:
                        # ⭐ 即使没有有效轮廓，也继续处理
                        largest_contour = None
                        points = np.array([]).reshape(0, 2)
                        points_global = np.array([]).reshape(0, 2)
                        
                        # 使用mask的质心作为中心点
                        M = cv2.moments(mask)
                        if M['m00'] != 0:
                            cx_roi = M['m10'] / M['m00']
                            cy_roi = M['m01'] / M['m00']
                            cx_global = cx_roi + x1
                            cy_global = cy_roi + y1
                        else:
                            cx_global = x1 + mask.shape[1] // 2
                            cy_global = y1 + mask.shape[0] // 2
                            cx_roi = cx_global - x1
                            cy_roi = cy_global - y1
                        
                        # 设置空的analysis stats
                        if analysis.get('stats') is None:
                            analysis['stats'] = {
                                'contour_roi': [],
                                'all_contours_roi': [],
                                'roi_offset': (x1, y1),
                                'center': (cx_global, cy_global),
                                'center_roi': (cx_roi, cy_roi),
                                'area': total_area,
                                'num_contours': 0
                            }
                    else:
                        # 选择最大轮廓用于计算质心和主显示
                        largest_contour = max(contours, key=cv2.contourArea)
                        points = largest_contour[:, 0, :]
                        points_global = points.copy()
                        points_global[:, 0] += x1
                        points_global[:, 1] += y1
                        
                        # 计算整体质心
                        M = cv2.moments(mask)
                        if M['m00'] != 0:
                            cx_roi = M['m10'] / M['m00']
                            cy_roi = M['m01'] / M['m00']
                            cx_global = cx_roi + x1
                            cy_global = cy_roi + y1
                        else:
                            cx_global = x1 + mask.shape[1] // 2
                            cy_global = y1 + mask.shape[0] // 2
                            cx_roi = cx_global - x1
                            cy_roi = cy_global - y1
                        
                        # 确保analysis['stats']被填充（在else分支中）
                        if analysis.get('stats') is None:
                            analysis['stats'] = {
                                'contour_roi': points.tolist() if len(points) > 0 else [],
                                'all_contours_roi': [pts.tolist() for pts in all_points_roi],
                                'roi_offset': (x1, y1),
                                'center': (cx_global, cy_global),
                                'center_roi': (cx_roi, cy_roi),
                                'area': total_area,
                                'num_contours': len(all_points_roi)
                            }
                
                # 记录多轮廓信息
                num_contours = len(all_points_roi) if 'all_points_roi' in locals() else 0
                total_contour_area_val = total_contour_area if 'total_contour_area' in locals() else 0
                if num_contours > 1 and z_idx % 20 == 0:
                    print(f"    🔗 z_idx={z_idx}: 3D连通的{num_contours}个2D区域 (总面积={total_contour_area_val}px)")
                
                # 调试：打印第一个轮廓点（仅在有效轮廓存在时）
                if z_idx == 30 and 'points' in locals() and len(points) > 0:
                    print(f"\n  🔍 调试轮廓 Z={z_idx}:")
                    print(f"    ROI: x1={x1}, x2={x2}, y1={y1}, y2={y2}")
                    print(f"    轮廓点数: {len(points)}")
                    print(f"    第1个轮廓点 - ROI局部: ({points[0, 0]}, {points[0, 1]})")
                    if 'points_global' in locals() and len(points_global) > 0:
                        print(f"    第1个轮廓点 - 全局: ({points_global[0, 0]}, {points_global[0, 1]})")
                    # 计算轮廓质心
                    if 'largest_contour' in locals() and largest_contour is not None:
                        M = cv2.moments(largest_contour)
                        if M['m00'] != 0:
                            cx_contour = M['m10'] / M['m00']
                            cy_contour = M['m01'] / M['m00']
                            print(f"    轮廓质心 - ROI局部: ({cx_contour:.1f}, {cy_contour:.1f})")
                            print(f"    轮廓质心 - 全局: ({cx_contour + x1:.1f}, {cy_contour + y1:.1f})")
                
                # 缩放到降采样空间(128×128)
                if self.original_size is not None and self.downsample_size is not None:
                    scale_factor = self.downsample_size / self.original_size[0]
                    points_scaled = points_global * scale_factor
                    
                    if z_idx == 30 and points_scaled is not None and len(points_scaled) > 0:
                        print(f"    第1个轮廓点 - 缩放: ({points_scaled[0, 0]:.1f}, {points_scaled[0, 1]:.1f})")
                        cx_scaled_contour = (cx_contour + x1) * scale_factor
                        cy_scaled_contour = (cy_contour + y1) * scale_factor
                        print(f"    轮廓质心 - 缩放: ({cx_scaled_contour:.1f}, {cy_scaled_contour:.1f})")
                        # 验证volume中的值
                        vol_x, vol_y = int(points_scaled[0, 0]), int(points_scaled[0, 1])
                        if 0 <= vol_x < 128 and 0 <= vol_y < 128 and z_idx < len(self.slices_data):
                            vol_val = self.volume[z_idx, vol_y, vol_x]
                            print(f"    Volume验证(轮廓点): [{z_idx}, {vol_y}, {vol_x}] = {vol_val:.3f}")
                        vol_x2, vol_y2 = int(cx_scaled_contour), int(cy_scaled_contour)
                        if 0 <= vol_x2 < 128 and 0 <= vol_y2 < 128:
                            vol_val2 = self.volume[z_idx, vol_y2, vol_x2]
                            print(f"    Volume验证(质心): [{z_idx}, {vol_y2}, {vol_x2}] = {vol_val2:.3f}")
                else:
                    points_scaled = points_global
                
                # 收集所有切片的轮廓和中心点数据（用于滑块显示）
                if (analysis and 'stats' in analysis and analysis['stats'] is not None 
                    and 'contour_roi' in analysis['stats']):
                    # 获取实际物理Z坐标(mm)
                    z_physical = self.slices_data[z_idx][0]
                    
                    # 使用步骤7保存的精确轮廓点（ROI坐标）
                    contour_roi_data = analysis['stats']['contour_roi']
                    if len(contour_roi_data) == 0:
                        # 空轮廓，创建空的二维数组
                        contour_points_roi = np.array([]).reshape(0, 2)
                    else:
                        contour_points_roi = np.array(contour_roi_data)  # [N, 2]
                        # 确保是二维数组
                        if contour_points_roi.ndim == 1:
                            contour_points_roi = contour_points_roi.reshape(-1, 2)
                    
                    roi_x1, roi_y1 = analysis['stats']['roi_offset']  # ROI在降采样空间中的偏移
                    
                    # 转换到降采样全局128×128坐标
                    if len(contour_points_roi) > 0:
                        contour_points_downsample = contour_points_roi.copy()
                        contour_points_downsample[:, 0] += roi_x1  # X坐标加列偏移
                        contour_points_downsample[:, 1] += roi_y1  # Y坐标加行偏移
                    else:
                        contour_points_downsample = np.array([]).reshape(0, 2)
                    
                    # 转换到原始512×512坐标
                    scale_up = self.original_size[0] / self.downsample_size
                    contour_points_original = contour_points_downsample * scale_up
                    
                    # 再缩放回降采样空间（为了与网格对齐）
                    scale_down = self.downsample_size / self.original_size[0]
                    contour_points_scaled = contour_points_original * scale_down
                    
                    # 中心点坐标
                    cx_original, cy_original = analysis['stats']['center']
                    cx_scaled = cx_original * scale_down
                    cy_scaled = cy_original * scale_down
                    
                    # 保存到列表
                    if len(contour_points_scaled) > 0:
                        contour_x_list = contour_points_scaled[:, 0].tolist()
                        contour_y_list = contour_points_scaled[:, 1].tolist()
                    else:
                        contour_x_list = []
                        contour_y_list = []
                    
                    all_slice_contours.append({
                        'z_idx': z_idx,
                        'z_physical': z_physical,
                        'contour_x': contour_x_list,
                        'contour_y': contour_y_list,
                        'center_x': cx_scaled,
                        'center_y': cy_scaled,
                        'area': analysis['stats']['area'],
                        'visible': False  # 默认不显示
                    })
                
                # 为所有有轮廓数据的切片添加到3D图中（通过滑块控制显示）
                if (analysis and 'stats' in analysis and analysis['stats'] is not None 
                    and 'contour_roi' in analysis['stats']):
                    # 获取实际物理Z坐标(mm)
                    z_physical = self.slices_data[z_idx][0]
                    
                    # 使用步骤7保存的精确轮廓点（ROI坐标，在降采样空间256×256中）
                    contour_roi_data = analysis['stats']['contour_roi']
                    if len(contour_roi_data) == 0:
                        # 空轮廓，创建空的二维数组
                        contour_points_roi = np.array([]).reshape(0, 2)
                    else:
                        contour_points_roi = np.array(contour_roi_data)  # [N, 2]
                        # 确保是二维数组
                        if contour_points_roi.ndim == 1:
                            contour_points_roi = contour_points_roi.reshape(-1, 2)
                    
                    roi_x1, roi_y1 = analysis['stats']['roi_offset']  # ROI在降采样空间中的偏移
                    
                    # ⭐ 简化坐标转换：ROI局部坐标 + 偏移 = 降采样全局坐标（与主网格一致）
                    # 不需要再做scale_up和scale_down，因为ROI已经在降采样空间中
                    if len(contour_points_roi) > 0:
                        contour_points_scaled = contour_points_roi.copy().astype(np.float64)
                        contour_points_scaled[:, 0] += roi_x1  # X坐标加列偏移
                        contour_points_scaled[:, 1] += roi_y1  # Y坐标加行偏移
                        
                        # 闭合轮廓
                        contour_points_closed = np.vstack([contour_points_scaled, contour_points_scaled[0:1, :]])
                    else:
                        # 空轮廓
                        contour_points_scaled = np.array([]).reshape(0, 2)
                        contour_points_closed = np.array([]).reshape(0, 2)
                    
                    # 创建3D坐标 - 使用实际物理Z坐标
                    if len(contour_points_closed) > 0:
                        x_coords = contour_points_closed[:, 0]
                        y_coords = contour_points_closed[:, 1]
                        z_coords = np.full(len(contour_points_closed), z_physical)
                    else:
                        # 空轮廓，创建空数组
                        x_coords = np.array([])
                        y_coords = np.array([])
                        z_coords = np.array([])
                    
                    if z_idx == 30:
                        print(f"    🔍 3D轮廓坐标（步骤7黄色轮廓）: Z_idx={z_idx}, Z_physical={z_physical:.1f}mm")
                        print(f"       步骤7轮廓点: {len(contour_points_roi)}个点")
                        print(f"       ROI偏移: ({roi_x1}, {roi_y1})")
                        if x_coords.size > 0 and y_coords.size > 0:
                            print(
                                f"       X范围[{x_coords.min():.1f}, {x_coords.max():.1f}], "
                                f"Y范围[{y_coords.min():.1f}, {y_coords.max():.1f}]"
                            )
                        else:
                            print("       X/Y范围: (空轮廓)")
                    
                    # 1. 添加CT图像平面（半透明，类似参考平面）
                    if z_idx < self.volume.shape[0]:
                        slice_2d = self.volume[z_idx]
                        # ⭐ 使用与主网格一致的尺寸（256×256）
                        grid_size = self.downsample_size  # 256
                        y_grid, x_grid = np.mgrid[0:grid_size, 0:grid_size]
                        z_grid = np.full((grid_size, grid_size), z_physical)
                        
                        fig.add_trace(go.Surface(
                            x=x_grid,
                            y=y_grid,
                            z=z_grid,
                            surfacecolor=slice_2d,
                            colorscale='gray',
                            showscale=False,
                            opacity=0.3,
                            visible=(z_idx == getattr(self, 'initial_display_z', 30)),
                            name=f'切片 Z={z_idx} CT图像 ({z_physical:.1f}mm)',
                            legendgroup=f'slice_{z_idx}',
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # 2. 添加轮廓线（黄色，与步骤7一致）
                    initial_z_for_display = getattr(self, 'initial_display_z', 30)
                    fig.add_trace(go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='lines',
                        line=dict(color='yellow', width=6),
                        visible=(z_idx == initial_z_for_display),
                        name=f'切片 Z={z_idx} 轮廓 ({z_physical:.1f}mm)',
                        legendgroup=f'slice_{z_idx}',
                        showlegend=(z_idx == initial_z_for_display)
                    ))
                    
                    # 3. 添加中心点（红色十字标记）
                    # ⭐ 使用ROI坐标+偏移 = 降采样全局坐标（与主网格一致）
                    if 'center_roi' in analysis['stats']:
                        cx_roi, cy_roi = analysis['stats']['center_roi']
                        cx_scaled = cx_roi + roi_x1
                        cy_scaled = cy_roi + roi_y1
                    else:
                        # 兼容旧格式：从原始坐标转换
                        cx_original, cy_original = analysis['stats']['center']
                        scale = self.downsample_size / self.original_size[0]
                        cx_scaled = cx_original * scale
                        cy_scaled = cy_original * scale
                    
                    if z_idx == 30:
                        print(f"    🔍 3D中心点: X={cx_scaled:.1f}, Y={cy_scaled:.1f}, Z={z_physical:.1f}mm")
                        print(f"       面积: {analysis['stats']['area']}px")
                    
                    fig.add_trace(go.Scatter3d(
                        x=[cx_scaled],
                        y=[cy_scaled],
                        z=[z_physical],
                        mode='markers',
                        marker=dict(size=12, color='red', symbol='x', line=dict(width=3)),
                        visible=(z_idx == initial_z_for_display),
                        name=f'切片 Z={z_idx} 中心 ({z_physical:.1f}mm)',
                        legendgroup=f'slice_{z_idx}',
                        showlegend=False
                    ))
                
                added_count += 1
                
                # 保存分析结果
                if analysis and 'images' in analysis:
                    self.cross_section_analysis.append(analysis)
                
                # ⭐ 确保所有切片都被添加到all_slice_contours中（即使没有找到合适的区域）
                z_physical = self.slices_data[z_idx][0]
                # 检查这个切片是否已经在all_slice_contours中
                if not any(s['z_idx'] == z_idx for s in all_slice_contours):
                    # 如果没有，添加一个空条目（至少包含基本信息）
                    all_slice_contours.append({
                        'z_idx': z_idx,
                        'z_physical': z_physical,
                        'contour_x': [],
                        'contour_y': [],
                        'center_x': 0,
                        'center_y': 0,
                        'area': 0,
                        'visible': False
                    })
            
            print(f"\n✓ 已处理 {added_count} 个横截面切片")
            print(f"✓ 生成了 {len(self.cross_section_analysis)} 个详细分析")
            print(f"✓ 收集了 {len(all_slice_contours)} 个切片的轮廓数据")
            
            # 保存所有切片数据供HTML使用
            self.all_slice_data = all_slice_contours
        
        # 5. (虚拟内窥镜功能已移除)
        # 6. (参考切片功能已整合到滑块控制中，每个切片都有CT图像、轮廓和中心点)
        
        # 布局设置
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                # orbit 与程序化 scene.camera 更新在部分浏览器上更不易出现「白屏直到点轨道旋转」的 GL 不同步
                dragmode='orbit',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)  # Z轴向上
                ),
                xaxis=dict(
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                    title="X (左右)",
                    visible=True
                ),
                yaxis=dict(
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                    title="Y (前后)",
                    visible=True
                ),
                zaxis=dict(
                    backgroundcolor="rgb(240, 240, 240)",
                    gridcolor="white",
                    showbackground=True,
                    title="Z (头足方向, mm)",
                    visible=True
                )
            ),
            width=1400,
            height=1000,
            title=dict(
                text=f"气管3D重建 (网格: {len(self.vertices):,}顶点)",
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            showlegend=True,
            legend=dict(x=0.02, y=0.98)
        )
        
        # 保存HTML (添加横截面分析和滑块控件)
        html_content = fig.to_html(include_plotlyjs='cdn')
        
        # 添加滑块控件和交互功能
        if hasattr(self, 'all_slice_data') and self.all_slice_data:
            slider_html = self._generate_slider_control()
            # 在</body>前插入滑块控件
            html_content = html_content.replace('</body>', slider_html + '</body>')
        
        # 在HTML中添加横截面分析部分
        if self.cross_section_analysis:
            analysis_html = self._generate_analysis_html()
            # 在</body>前插入分析内容
            html_content = html_content.replace('</body>', analysis_html + '</body>')

        # 在HTML最底部添加实验信息（参数/耗时/简介）
        meta_html = self._generate_experiment_meta_html(ended_at=datetime.datetime.now())
        html_content = html_content.replace('</body>', meta_html + '</body>')

        # 在HTML中嵌入导航线点列 + 沿导航线绑定相机（首版：播放/暂停/重置）
        if navigation_line and self.navigation_path_plotly is not None:
            try:
                nav_payload = {
                    "points_plotly": self.navigation_path_plotly.tolist(),
                    "meta": self.navigation_meta or {},
                }
                nav_script = (
                    "<script>\n"
                    "window.__AIRWAY_NAVIGATION__ = "
                    + json.dumps(nav_payload, ensure_ascii=False)
                    + ";\n"
                    "</script>\n"
                )
                nav_cam_html = self._generate_navigation_camera_binding_html()
                html_content = html_content.replace('</body>', nav_script + nav_cam_html + '</body>')
            except Exception:
                pass
        
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        file_size_mb = os.path.getsize(output_html) / 1024 / 1024
        print(f"\n✓ 可视化已保存: {output_html}")
        print(f"  文件大小: {file_size_mb:.1f} MB")
        
        return fig

    def _generate_navigation_camera_binding_html(self) -> str:
        """在 HTML 中注入沿导航线更新 Plotly scene.camera 的控件（读取 window.__AIRWAY_NAVIGATION__）。"""
        return """
        <div id='navCamPanel' style='position:fixed;bottom:20px;right:20px;background:#0b1220;color:#e5e7eb;padding:10px 12px;border-radius:12px;box-shadow:0 2px 14px rgba(0,0,0,0.42);z-index:9998;width:340px;font-family:system-ui,-apple-system,sans-serif;font-size:13px;'>
            <div id='navCamHeader' style='display:flex;align-items:center;justify-content:space-between;gap:10px;cursor:pointer;user-select:none;'>
                <div style='font-weight:700;color:#fff;letter-spacing:0.2px;'>导航线相机漫游</div>
                <button id='navCamCollapse' type='button' title='折叠/展开' style='padding:4px 8px;border-radius:8px;border:1px solid #334155;background:#111827;color:#e5e7eb;cursor:pointer;font-size:12px;line-height:1;'>收起</button>
            </div>
            <div id='navCamBody' style='margin-top:10px;'>
                <div style='display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px;'>
                    <button id='navCamPlay' type='button' style='flex:1;padding:6px 8px;border-radius:8px;border:1px solid #334155;background:#2563eb;color:#fff;cursor:pointer;font-size:12px;font-weight:650;'>播放</button>
                    <button id='navCamPause' type='button' style='flex:1;padding:6px 8px;border-radius:8px;border:1px solid #334155;background:#111827;color:#e5e7eb;cursor:pointer;font-size:12px;'>暂停</button>
                    <button id='navCamReset' type='button' style='flex:1;padding:6px 8px;border-radius:8px;border:1px solid #334155;background:#111827;color:#e5e7eb;cursor:pointer;font-size:12px;'>重置视角</button>
                </div>
                <div style='margin:-2px 0 10px;font-size:11px;color:#94a3b8;line-height:1.45;'>播放时：相机<b>严格在导航线当前顶点</b>；对<b>视线方向</b>做一阶混合平滑（拖滑跳帧会立刻对齐 raw）；参考路径<b>前方第 3 个顶点</b>定 raw 方向；点<b>暂停</b>后可自由旋转、缩放。<br/>图例：<b>相机位置</b>（粉菱形）、<b>视线</b>（黄线段 eye→注视点）、<b>注视点</b>（橙十字）与 Plotly 相机一致。<br/><span style='color:#cbd5e1;'>更贴管腔：提高下方 JS 中 <code>navFwdFollow</code>（越接近 1 越跟折线、弯折处可能略抖）。</span></div>
                <button id='navLumToggle' type='button' style='width:100%;margin-bottom:10px;padding:8px 10px;border-radius:10px;border:1px solid #334155;background:#14532d;color:#ecfdf5;cursor:pointer;font-size:12px;font-weight:700;'>腔内模式：关</button>
                <div id='navLumState' style='display:none;margin:-4px 0 10px;font-size:11px;color:#a7f3d0;line-height:1.35;'></div>
                <div style='margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;font-size:12px;color:#cbd5e1;'>
                        <span>路径帧 <b id='navCamIdx'>0</b> / <b id='navCamIdxMax'>0</b></span>
                        <span>Z <b id='navCamZmm'>—</b> mm</span>
                    </div>
                    <input id='navCamPath' type='range' min='0' max='0' value='0' step='1' style='width:100%;' />
                    <div style='margin-top:2px;font-size:11px;color:#64748b;'>顺序与导航线一致（头侧 zmax → zmin）；拖动选帧会自动暂停播放。</div>
                </div>
                <label style='display:flex;align-items:center;gap:10px;font-size:12px;color:#cbd5e1;'>
                    步进间隔 (ms)
                    <input id='navCamMs' type='range' min='40' max='280' value='110' style='flex:1;' />
                    <span id='navCamMsLbl'>110</span>
                </label>
                <div style='margin-top:6px;font-size:11px;color:#94a3b8;line-height:1.35;'>
                    掩码体素 1 = <b>气腔（充气连通区域）</b>，不是软组织把心腔填实；MC 网格是包绕气腔的壳，相机在腔内时外表面常为背面。<br/>
                    点<b>「腔内模式」</b>：隐藏「其他结构/气管外表面」，显示<b>反向法线内壁</b>并加粗导航线，便于腔内看壁形态。
                </div>
            </div>
        </div>
        <script>
        (function() {
            function applyCollapsed(collapsed) {
                var body = document.getElementById('navCamBody');
                var btn = document.getElementById('navCamCollapse');
                if (!body || !btn) return;
                body.style.display = collapsed ? 'none' : 'block';
                btn.textContent = collapsed ? '展开' : '收起';
                try { localStorage.setItem('__NAV_CAM_COLLAPSED__', collapsed ? '1' : '0'); } catch (e) {}
            }
            function getCollapsed() {
                try { return localStorage.getItem('__NAV_CAM_COLLAPSED__') === '1'; } catch (e) { return false; }
            }
            function getPts() {
                try {
                    var nav = window.__AIRWAY_NAVIGATION__;
                    if (!nav || !nav.points_plotly || nav.points_plotly.length < 2) return null;
                    return nav.points_plotly;
                } catch (e) { return null; }
            }
            function getGd() {
                return document.querySelector('.plotly-graph-div');
            }
            function cloneCam(cam) {
                if (!cam) return null;
                return {
                    up: cam.up ? {x: cam.up.x, y: cam.up.y, z: cam.up.z} : {x: 0, y: 0, z: 1},
                    center: cam.center ? {x: cam.center.x, y: cam.center.y, z: cam.center.z} : {x: 0, y: 0, z: 0},
                    eye: cam.eye ? {x: cam.eye.x, y: cam.eye.y, z: cam.eye.z} : {x: 1.25, y: 1.25, z: 1.25}
                };
            }
            var timer = null;
            var idx = 0;
            var initialCam = null;
            var lookAhead = 3;
            var luminalOn = false;
            // 漫游：eye 始终在路径 pts[ii]；视线用 blendDir 在 rawFwd 上低通，减轻抖晃。
            // navFwdFollow：每帧朝 rawFwd 靠拢的比例；越大越贴真实折线/管腔走向（窄弯更准），顶点处方向跳变更明显→可能略抖。
            var navFwdFollow = 0.42;
            var navSmFwd = null;
            var navLastIi = null;

            function resetNavCameraSmooth() {
                navSmFwd = null;
                navLastIi = null;
            }

            function blendDir(d0, d1, a) {
                var x = (1 - a) * d0[0] + a * d1[0];
                var y = (1 - a) * d0[1] + a * d1[1];
                var z = (1 - a) * d0[2] + a * d1[2];
                return norm3([x, y, z]);
            }

            function traceIdxByName(gd, name) {
                if (!gd || !gd.data) return null;
                for (var t = 0; t < gd.data.length; t++) {
                    if (gd.data[t] && gd.data[t].name === name) return t;
                }
                return null;
            }
            function restyleVis(gd, ti, vis) {
                if (ti == null || ti < 0) return;
                Plotly.restyle(gd, { visible: vis }, [ti]);
            }
            function updateNavDebugOverlays(gd, ex, ey, ez, cx, cy, cz) {
                if (!gd) return;
                if (!isFinite(ex) || !isFinite(ey) || !isFinite(ez)) return;
                if (!isFinite(cx) || !isFinite(cy) || !isFinite(cz)) return;
                var tiE = traceIdxByName(gd, '相机位置(漫游)');
                var tiG = traceIdxByName(gd, '视线(漫游)');
                var tiA = traceIdxByName(gd, '注视点(漫游)');
                try {
                    if (tiE != null) Plotly.restyle(gd, { x: [[ex]], y: [[ey]], z: [[ez]] }, [tiE]);
                    if (tiA != null) Plotly.restyle(gd, { x: [[cx]], y: [[cy]], z: [[cz]] }, [tiA]);
                    if (tiG != null) Plotly.restyle(gd, { x: [[ex, cx]], y: [[ey, cy]], z: [[ez, cz]] }, [tiG]);
                } catch (e) {}
            }
            function applyLuminalMode(on) {
                luminalOn = !!on;
                var gd = getGd();
                var st = document.getElementById('navLumState');
                if (!gd || !gd.data) {
                    if (st) { st.style.display = 'none'; }
                    return;
                }
                var bg = traceIdxByName(gd, '其他结构(半透明)');
                var outer = traceIdxByName(gd, '气管区域(3D充气法)');
                var inner = traceIdxByName(gd, '气管内壁(腔内)');
                var nav = traceIdxByName(gd, '导航线(插管路径)');
                var cl = traceIdxByName(gd, '充气中心线');
                var camM = traceIdxByName(gd, '相机位置(漫游)');
                var tiGaze = traceIdxByName(gd, '视线(漫游)');
                var tiAt = traceIdxByName(gd, '注视点(漫游)');
                if (luminalOn) {
                    restyleVis(gd, bg, false);
                    restyleVis(gd, outer, false);
                    if (inner != null) {
                        restyleVis(gd, inner, true);
                        Plotly.restyle(gd, { opacity: 0.58 }, [inner]);
                    }
                    restyleVis(gd, nav, true);
                    restyleVis(gd, cl, true);
                    if (camM != null) restyleVis(gd, camM, true);
                    if (tiGaze != null) restyleVis(gd, tiGaze, true);
                    if (tiAt != null) restyleVis(gd, tiAt, true);
                    if (nav != null) Plotly.restyle(gd, { 'line.width': 14, 'marker.size': 8 }, [nav]);
                    if (st) {
                        st.style.display = 'block';
                        st.textContent = inner != null
                            ? '腔内模式：已显示反向法线内壁；外壁与背景已隐藏。'
                            : '腔内模式：未找到内壁图层（请用最新脚本重导 HTML）。';
                    }
                } else {
                    restyleVis(gd, bg, true);
                    restyleVis(gd, outer, true);
                    if (inner != null) {
                        restyleVis(gd, inner, false);
                        Plotly.restyle(gd, { opacity: 0.48 }, [inner]);
                    }
                    restyleVis(gd, nav, true);
                    restyleVis(gd, cl, true);
                    if (camM != null) restyleVis(gd, camM, true);
                    if (tiGaze != null) restyleVis(gd, tiGaze, true);
                    if (tiAt != null) restyleVis(gd, tiAt, true);
                    if (nav != null) Plotly.restyle(gd, { 'line.width': 8, 'marker.size': 4 }, [nav]);
                    if (st) { st.style.display = 'none'; }
                }
                var b = document.getElementById('navLumToggle');
                if (b) b.textContent = luminalOn ? '腔内模式：开（再点关闭）' : '腔内模式：关';
            }

            function pickUp(fx, fy, fz) {
                var ux = 0, uy = 0, uz = 1;
                var nx = fx, ny = fy, nz = fz;
                var fn = Math.sqrt(nx * nx + ny * ny + nz * nz);
                if (fn > 1e-9) { nx /= fn; ny /= fn; nz /= fn; }
                var dot = Math.abs(nx * ux + ny * uy + nz * uz);
                if (dot > 0.92) { ux = 0; uy = 1; uz = 0; }
                return {x: ux, y: uy, z: uz};
            }

            function norm3(v) {
                var l = Math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
                if (l < 1e-12) return [0, 0, 1];
                return [v[0]/l, v[1]/l, v[2]/l];
            }

            function tangentForward(pts, ii) {
                var n = pts.length;
                if (n < 2) return [0, 0, 1];
                if (ii < n - 1) {
                    return [pts[ii+1][0]-pts[ii][0], pts[ii+1][1]-pts[ii][1], pts[ii+1][2]-pts[ii][2]];
                }
                return [pts[n-1][0]-pts[n-2][0], pts[n-1][1]-pts[n-2][1], pts[n-1][2]-pts[n-2][2]];
            }

            // 用多段中心差分估计切向，避免「相邻两点差分」在密折线上的高频抖动
            function tangentSmoothed(pts, ii, halfWin) {
                halfWin = (halfWin == null) ? 5 : halfWin;
                var n = pts.length;
                if (n < 2) return [0, 0, 1];
                var i0 = Math.max(0, ii - halfWin);
                var i1 = Math.min(n - 1, ii + halfWin);
                if (i0 === i1) {
                    return norm3(tangentForward(pts, ii));
                }
                var vx = pts[i1][0] - pts[i0][0];
                var vy = pts[i1][1] - pts[i0][1];
                var vz = pts[i1][2] - pts[i0][2];
                return norm3([vx, vy, vz]);
            }

            function meanStep(pts) {
                var lim = Math.min(pts.length - 1, 80);
                var acc = 0, cnt = 0;
                for (var k = 0; k < lim; k++) {
                    var dx = pts[k+1][0]-pts[k][0], dy = pts[k+1][1]-pts[k][1], dz = pts[k+1][2]-pts[k][2];
                    acc += Math.sqrt(dx*dx + dy*dy + dz*dz);
                    cnt++;
                }
                return cnt ? (acc / cnt) : 1.0;
            }

            function deepMergeCam(oldCam, patch) {
                try {
                    var o = oldCam ? JSON.parse(JSON.stringify(oldCam)) : {};
                    for (var k in patch) { if (Object.prototype.hasOwnProperty.call(patch, k)) o[k] = patch[k]; }
                    return o;
                } catch (e) {
                    return patch;
                }
            }

            // 程序化 relayout(scene.camera) 后，部分环境下 WebGL 子图不立刻重绘（白屏），点模式栏「轨道旋转」会触发刷新。
            // 用 resize / rAF 再触发一次布局，促使 GL 与 layout 同步（见 plotly.js 相关 issue 讨论）。
            function forceSceneRedraw(gd) {
                if (!gd) return;
                function one() {
                    try {
                        if (typeof Plotly !== 'undefined' && Plotly.Plots && typeof Plotly.Plots.resize === 'function') {
                            Plotly.Plots.resize(gd);
                        }
                    } catch (e0) {}
                }
                one();
                try {
                    if (typeof requestAnimationFrame === 'function') {
                        requestAnimationFrame(function() { one(); });
                    }
                } catch (e1) {}
            }

            function relayoutSceneCameraPatch(gd, patch) {
                if (!gd || !patch) return;
                try {
                    var p = Plotly.relayout(gd, patch);
                    if (p && typeof p.then === 'function') {
                        p.then(function() { forceSceneRedraw(gd); });
                    } else {
                        setTimeout(function() { forceSceneRedraw(gd); }, 0);
                    }
                } catch (e) {
                    setTimeout(function() { forceSceneRedraw(gd); }, 0);
                }
            }

            // Plotly scene.camera 的 eye/center/up 在「场景域坐标」中：(0,0,0) 为数据包围盒中心，
            // 量级约与默认 eye=(1.25,1.25,1.25) 相当；不能把导航点的数据坐标(mm/像素)直接写入，否则会白屏。
            function getSceneCamBasis(gd) {
                var L = gd._fullLayout || gd.layout;
                if (!L || !L.scene) return null;
                var sc = L.scene;
                var xr = sc.xaxis && sc.xaxis.range;
                var yr = sc.yaxis && sc.yaxis.range;
                var zr = sc.zaxis && sc.zaxis.range;
                if (!xr || xr.length < 2 || !yr || yr.length < 2 || !zr || zr.length < 2) return null;
                var x0 = Math.min(xr[0], xr[1]), x1 = Math.max(xr[0], xr[1]);
                var y0 = Math.min(yr[0], yr[1]), y1 = Math.max(yr[0], yr[1]);
                var z0 = Math.min(zr[0], zr[1]), z1 = Math.max(zr[0], zr[1]);
                var cx = (x0 + x1) * 0.5, cy = (y0 + y1) * 0.5, cz = (z0 + z1) * 0.5;
                var hx = Math.max(1e-12, (x1 - x0) * 0.5);
                var hy = Math.max(1e-12, (y1 - y0) * 0.5);
                var hz = Math.max(1e-12, (z1 - z0) * 0.5);
                var ar = sc.aspectratio || {x: 1, y: 1, z: 1};
                var ax = (ar.x != null) ? ar.x : 1;
                var ay = (ar.y != null) ? ar.y : 1;
                var az = (ar.z != null) ? ar.z : 1;
                return { cx: cx, cy: cy, cz: cz, hx: hx, hy: hy, hz: hz, ax: ax, ay: ay, az: az };
            }
            function dataPointToSceneCam(pt, b) {
                if (!b) return null;
                return {
                    x: 0.5 * b.ax * (pt[0] - b.cx) / b.hx,
                    y: 0.5 * b.ay * (pt[1] - b.cy) / b.hy,
                    z: 0.5 * b.az * (pt[2] - b.cz) / b.hz
                };
            }

            function getSceneCamBasisFromTraces(gd) {
                if (!gd || !gd.data) return null;
                var xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity, zmin = Infinity, zmax = -Infinity;
                function accArr(xa, ya, za) {
                    if (!xa || !ya || !za) return;
                    var n = Math.min(xa.length, ya.length, za.length);
                    for (var k = 0; k < n; k++) {
                        var x = xa[k], y = ya[k], z = za[k];
                        if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
                        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
                        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
                        if (z < zmin) zmin = z; if (z > zmax) zmax = z;
                    }
                }
                for (var t = 0; t < gd.data.length; t++) {
                    var tr = gd.data[t];
                    if (!tr) continue;
                    if (tr.type === 'scatter3d' || tr.type === 'mesh3d') {
                        accArr(tr.x, tr.y, tr.z);
                    }
                }
                if (!isFinite(xmin) || !isFinite(xmax)) return null;
                var cx = (xmin + xmax) * 0.5, cy = (ymin + ymax) * 0.5, cz = (zmin + zmax) * 0.5;
                var hx = Math.max(1e-12, (xmax - xmin) * 0.5);
                var hy = Math.max(1e-12, (ymax - ymin) * 0.5);
                var hz = Math.max(1e-12, (zmax - zmin) * 0.5);
                return { cx: cx, cy: cy, cz: cz, hx: hx, hy: hy, hz: hz, ax: 1, ay: 1, az: 1 };
            }

            function getSceneCamBasisAny(gd) {
                var b = getSceneCamBasis(gd);
                if (b) return b;
                return getSceneCamBasisFromTraces(gd);
            }

            function applyCameraForIndex(pts, i, skipRelayout) {
                var gd = getGd();
                if (!gd || !pts || pts.length < 2) return;
                var n = pts.length;
                var ii = Math.max(0, Math.min(n - 1, i | 0));
                var step = meanStep(pts);
                var lookLen = Math.max(step * 2.5, 1.2);
                var lookSegs = 3;
                var j = Math.min(n - 1, ii + lookSegs);
                var rawEx = pts[ii][0];
                var rawEy = pts[ii][1];
                var rawEz = pts[ii][2];
                var tx = pts[j][0];
                var ty = pts[j][1];
                var tz = pts[j][2];
                var rdx = tx - rawEx;
                var rdy = ty - rawEy;
                var rdz = tz - rawEz;
                var rdist = Math.sqrt(rdx * rdx + rdy * rdy + rdz * rdz);
                if (rdist < 1e-4) {
                    var fwb = tangentSmoothed(pts, Math.max(0, ii - 1), 8);
                    tx = rawEx + fwb[0] * lookLen;
                    ty = rawEy + fwb[1] * lookLen;
                    tz = rawEz + fwb[2] * lookLen;
                    rdx = tx - rawEx;
                    rdy = ty - rawEy;
                    rdz = tz - rawEz;
                }
                var rawFwd = norm3([tx - rawEx, ty - rawEy, tz - rawEz]);
                var jump = (navLastIi == null || Math.abs(ii - navLastIi) > 2);
                if (navSmFwd == null || jump) {
                    navSmFwd = rawFwd.slice();
                } else {
                    navSmFwd = blendDir(navSmFwd, rawFwd, navFwdFollow);
                }
                navLastIi = ii;
                var ex = rawEx;
                var ey = rawEy;
                var ez = rawEz;
                var lookDist = Math.max(step * 3.5, 2.0);
                var cx = ex + navSmFwd[0] * lookDist;
                var cy = ey + navSmFwd[1] * lookDist;
                var cz = ez + navSmFwd[2] * lookDist;
                updateNavDebugOverlays(gd, ex, ey, ez, cx, cy, cz);
                if (skipRelayout) return;
                var b = getSceneCamBasisAny(gd);
                if (!b) return;
                var eyeC = dataPointToSceneCam([ex, ey, ez], b);
                var centerC = dataPointToSceneCam([cx, cy, cz], b);
                if (!eyeC || !centerC) return;
                if (!isFinite(eyeC.x) || !isFinite(centerC.x)) return;
                var fxc = centerC.x - eyeC.x, fyc = centerC.y - eyeC.y, fzc = centerC.z - eyeC.z;
                var up = pickUp(fxc, fyc, fzc);
                var oldCam = (gd.layout && gd.layout.scene && gd.layout.scene.camera) ? gd.layout.scene.camera : {};
                var patch = {
                    'scene.camera.eye': { x: eyeC.x, y: eyeC.y, z: eyeC.z },
                    'scene.camera.center': { x: centerC.x, y: centerC.y, z: centerC.z },
                    'scene.camera.up': { x: up.x, y: up.y, z: up.z }
                };
                try {
                    if (oldCam && oldCam.projection) {
                        patch['scene.camera.projection'] = JSON.parse(JSON.stringify(oldCam.projection));
                    }
                } catch (ep) {}
                relayoutSceneCameraPatch(gd, patch);
            }

            function syncPathUi(pts, i) {
                var path = document.getElementById('navCamPath');
                var iLbl = document.getElementById('navCamIdx');
                var mLbl = document.getElementById('navCamIdxMax');
                var zLbl = document.getElementById('navCamZmm');
                if (!pts || pts.length < 1) return;
                var n = pts.length;
                var ii = Math.max(0, Math.min(n - 1, i | 0));
                if (path) {
                    path.min = '0';
                    path.max = String(Math.max(0, n - 1));
                    path.value = String(ii);
                }
                if (iLbl) iLbl.textContent = String(ii);
                if (mLbl) mLbl.textContent = String(Math.max(0, n - 1));
                if (zLbl) {
                    var z = pts[ii][2];
                    zLbl.textContent = (z != null && isFinite(z)) ? Number(z).toFixed(1) : '—';
                }
            }

            function stop() {
                if (timer) { clearInterval(timer); timer = null; }
            }

            function start() {
                var pts = getPts();
                if (!pts) return;
                stop();
                var msInput = document.getElementById('navCamMs');
                var ms = msInput ? parseInt(msInput.value, 10) : 80;
                if (!isFinite(ms) || ms < 10) ms = 110;
                timer = setInterval(function() {
                    var pts2 = getPts();
                    if (!pts2) { stop(); return; }
                    applyCameraForIndex(pts2, idx);
                    syncPathUi(pts2, idx);
                    idx += 1;
                    if (idx >= pts2.length) idx = 0;
                }, ms);
            }

            function resetCam() {
                stop();
                resetNavCameraSmooth();
                var gd = getGd();
                var pts = getPts();
                idx = 0;
                if (pts) syncPathUi(pts, 0);
                if (pts && gd) applyCameraForIndex(pts, 0, true);
                if (!gd || !initialCam) return;
                relayoutSceneCameraPatch(gd, { 'scene.camera': initialCam });
            }

            function wire() {
                // 右下角面板：可折叠
                var header = document.getElementById('navCamHeader');
                var col = document.getElementById('navCamCollapse');
                var collapsed = getCollapsed();
                applyCollapsed(collapsed);
                function toggleCollapsed(ev) {
                    try { if (ev && ev.preventDefault) ev.preventDefault(); } catch (e) {}
                    collapsed = !collapsed;
                    applyCollapsed(collapsed);
                }
                if (col) col.addEventListener('click', function(ev) { toggleCollapsed(ev); });
                if (header) header.addEventListener('click', function(ev) {
                    // 点击标题栏也可折叠；但点按钮会冒泡到 header，因此这里忽略按钮本身
                    if (ev && ev.target && ev.target.id === 'navCamCollapse') return;
                    toggleCollapsed(ev);
                });
                var play = document.getElementById('navCamPlay');
                var pause = document.getElementById('navCamPause');
                var reset = document.getElementById('navCamReset');
                var ms = document.getElementById('navCamMs');
                var msLbl = document.getElementById('navCamMsLbl');
                var path = document.getElementById('navCamPath');
                if (play) play.addEventListener('click', start);
                if (pause) pause.addEventListener('click', stop);
                if (reset) reset.addEventListener('click', resetCam);
                if (ms && msLbl) {
                    ms.addEventListener('input', function() { msLbl.textContent = String(ms.value); });
                }
                if (path) {
                    path.addEventListener('input', function() {
                        var pts = getPts();
                        if (!pts) return;
                        stop();
                        idx = parseInt(path.value, 10) || 0;
                        idx = Math.max(0, Math.min(pts.length - 1, idx));
                        applyCameraForIndex(pts, idx);
                        syncPathUi(pts, idx);
                    });
                }
                var lumBtn = document.getElementById('navLumToggle');
                if (lumBtn) lumBtn.addEventListener('click', function() { applyLuminalMode(!luminalOn); });
            }

            function captureInitial() {
                var gd = getGd();
                if (!gd || !gd.layout || !gd.layout.scene || !gd.layout.scene.camera) return;
                if (!initialCam) initialCam = cloneCam(gd.layout.scene.camera);
            }

            function boot() {
                var pts = getPts();
                if (!pts) return;
                lookAhead = Math.max(1, Math.floor(pts.length * 0.03));
                idx = 0;
                syncPathUi(pts, 0);
                var gd = getGd();
                var capDone = false;
                function onAfter() {
                    if (capDone) return;
                    captureInitial();
                    if (initialCam) {
                        capDone = true;
                        forceSceneRedraw(getGd());
                    }
                }
                if (gd && gd.on) {
                    gd.on('plotly_afterplot', onAfter);
                }
                setTimeout(function() {
                    captureInitial();
                    forceSceneRedraw(getGd());
                    var g2 = getGd();
                    var p2 = getPts();
                    if (g2 && p2 && p2.length) applyCameraForIndex(p2, 0, true);
                }, 300);
                wire();
            }

            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', boot);
            } else {
                boot();
            }
        })();
        </script>
        """
    
    def _generate_slider_control(self):
        """生成切片滑块控件的HTML和JavaScript代码 - 全新重写版本"""
        import json
        
        # 准备切片数据：只包含必要信息（确保转换为Python原生类型）
        slice_list = []
        for s in self.all_slice_data:
            slice_list.append({
                'z_idx': int(s['z_idx']),
                'z_mm': round(float(s['z_physical']), 1),
                'area': int(s['area'])
            })
        
        # 选择初始切片：优先“种子切片”，否则30，否则中间
        seed_z_idx = getattr(self, "seed_z_idx", None)
        if seed_z_idx is not None and any(s['z_idx'] == int(seed_z_idx) for s in slice_list):
            initial_z_idx = int(seed_z_idx)
        else:
            initial_z_idx = 30 if any(s['z_idx'] == 30 for s in slice_list) else slice_list[len(slice_list)//2]['z_idx']
        self.initial_display_z = initial_z_idx  # 保存供trace使用
        
        # 找到初始切片的数组索引
        initial_array_index = 0
        for i, s in enumerate(slice_list):
            if s['z_idx'] == initial_z_idx:
                initial_array_index = i
                break
        
        # 获取初始切片的z_mm值用于显示
        initial_slice = slice_list[initial_array_index]
        
        # 生成HTML和JavaScript
        html = f"""
        <div id="slicePanel" style="position:fixed;top:20px;right:20px;background:#fff;padding:15px;border-radius:8px;box-shadow:0 2px 12px rgba(0,0,0,0.15);z-index:9999;width:280px;font-family:system-ui,-apple-system,sans-serif;">
            <h3 style="margin:0 0 12px;font-size:16px;color:#333;">切片浏览器</h3>
            <div style="margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
                    <span style="font-size:13px;color:#666;">切片 #<b id="zIdx">{initial_z_idx}</b></span>
                    <span style="font-size:13px;color:#666;">Z: <b id="zMm">{initial_slice['z_mm']}</b>mm</span>
                </div>
                <input type="range" id="slider" min="0" max="{len(slice_list)-1}" value="{initial_array_index}" style="width:100%;margin:8px 0;">
                <div style="display:flex;gap:8px;margin-top:8px;">
                    <button id="btnPrev" style="flex:1;padding:6px;border:1px solid #ddd;background:#f8f9fa;border-radius:4px;cursor:pointer;font-size:12px;">← 上一个</button>
                    <button id="btnNext" style="flex:1;padding:6px;border:1px solid #ddd;background:#f8f9fa;border-radius:4px;cursor:pointer;font-size:12px;">下一个 →</button>
                </div>
            </div>
            <div style="background:#f8f9fa;padding:10px;border-radius:4px;font-size:12px;">
                <div style="margin-bottom:4px;">面积: <b id="area">-</b> px</div>
                <div>Z物理: <b id="zMm2">-</b> mm</div>
            </div>
        </div>
        
        <script>
        (function() {{
            // 切片数据（简化版）
            const SLICES = {json.dumps(slice_list)};
            const INITIAL_Z = {initial_z_idx};
            const INITIAL_ARRAY_INDEX = {initial_array_index};
            
            // DOM元素
            const slider = document.getElementById('slider');
            const zIdx = document.getElementById('zIdx');
            const zMm = document.getElementById('zMm');
            const area = document.getElementById('area');
            const zMm2 = document.getElementById('zMm2');
            const btnPrev = document.getElementById('btnPrev');
            const btnNext = document.getElementById('btnNext');
            
            let currentArrayIndex = 0;
            let plotlyReady = false;
            
            // 更新UI和3D视图
            function showSlice(arrayIndex) {{
                if (arrayIndex < 0 || arrayIndex >= SLICES.length) return;
                
                const slice = SLICES[arrayIndex];
                currentArrayIndex = arrayIndex;
                
                // 更新UI
                slider.value = arrayIndex;
                zIdx.textContent = slice.z_idx;
                zMm.textContent = slice.z_mm;
                area.textContent = slice.area;
                zMm2.textContent = slice.z_mm;
                
                // 更新按钮状态
                btnPrev.disabled = (arrayIndex === 0);
                btnNext.disabled = (arrayIndex === SLICES.length - 1);
                
                // ⭐ 切换显示切片分析：隐藏所有，只显示当前
                const allSliceDivs = document.querySelectorAll('.slice-analysis-div');
                allSliceDivs.forEach(div => {{
                    div.style.display = 'none';
                }});
                const currentSliceDiv = document.getElementById('slice-analysis-' + slice.z_idx);
                if (currentSliceDiv) {{
                    currentSliceDiv.style.display = 'block';
                    // 滚动到该切片位置
                    currentSliceDiv.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
                }}
                
                // 更新Plotly可见性
                if (!plotlyReady) return;
                
                const gd = document.querySelector('.plotly-graph-div');
                if (!gd || !gd.data) return;
                
                // 构建可见性数组：只显示当前z_idx的切片trace
                const visibility = gd.data.map(trace => {{
                    if (!trace.name || !trace.name.includes('切片 Z=')) {{
                        return trace.visible; // 保持原状
                    }}
                    const match = trace.name.match(/切片 Z=(\\d+)/);
                    return match ? (parseInt(match[1]) === slice.z_idx) : false;
                }});
                
                Plotly.restyle(gd, {{'visible': visibility}});
            }}
            
            // 查找初始切片的数组索引
            function findInitialIndex() {{
                for (let i = 0; i < SLICES.length; i++) {{
                    if (SLICES[i].z_idx === INITIAL_Z) return i;
                }}
                return Math.floor(SLICES.length / 2);
            }}
            
            // 事件绑定
            slider.addEventListener('input', () => showSlice(parseInt(slider.value)));
            btnPrev.addEventListener('click', () => showSlice(currentArrayIndex - 1));
            btnNext.addEventListener('click', () => showSlice(currentArrayIndex + 1));
            
            // 键盘快捷键
            document.addEventListener('keydown', e => {{
                if (e.key === 'ArrowLeft') btnPrev.click();
                else if (e.key === 'ArrowRight') btnNext.click();
            }});
            
            // 等待Plotly加载完成
            function init() {{
                const gd = document.querySelector('.plotly-graph-div');
                if (gd && gd.data && gd.data.length > 0) {{
                    plotlyReady = true;
                    showSlice(INITIAL_ARRAY_INDEX);
                    console.log('✓ 切片选择器已初始化:', SLICES.length, '个切片');
                }} else {{
                    setTimeout(init, 50);
                }}
            }}
            
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', init);
            }} else {{
                init();
            }}
        }})();
        </script>
        """
        
        return html
    
    def _generate_analysis_html(self):
        """生成横截面和中心线分析的HTML内容"""
        html = """
        <div style="padding: 20px; background-color: #f5f5f5; margin-top: 20px;">
            <h2 style="text-align: center; color: #333;">气管分析详细报告</h2>
            <p style="text-align: center; color: #666;">完整的算法处理过程展示</p>
        """
        
        # 添加横截面分析部分
        if hasattr(self, 'cross_section_analysis') and len(self.cross_section_analysis) > 0:
            if self.fixed_threshold is not None:
                threshold_desc = (
                    f"固定阈值：{self.fixed_threshold:g}（窗位窗宽归一化[0,1]；WC=-600, WW=1500）"
                )
            else:
                threshold_desc = f"分位阈值：{self.percentile:g}% 分位（每切片自适应）"
            html += """
            <div style="margin: 30px 0;">
                <h2 style="color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px;">
                    🔍 横截面轮廓分析
                </h2>
                <p style="color: #666;">气管横截面的完整轮廓提取过程（当前二值化策略：""" + threshold_desc + """）</p>
            </div>
            """
        
        # 确定初始显示的切片z_idx（与滑块保持一致）
        initial_z_idx = None
        if hasattr(self, 'initial_display_z'):
            initial_z_idx = self.initial_display_z
        elif hasattr(self, 'all_slice_data') and self.all_slice_data:
            # 优先30，否则中间
            initial_z_idx = 30 if any(s['z_idx'] == 30 for s in self.all_slice_data) else self.all_slice_data[len(self.all_slice_data)//2]['z_idx']
        else:
            # 默认显示第一个
            initial_z_idx = self.cross_section_analysis[0]['z_index'] if self.cross_section_analysis else None
        
        for analysis in self.cross_section_analysis:
            z_idx = analysis['z_index']
            images = analysis.get('images', {})
            stats = analysis.get('stats', {})
            
            # 给每个切片div添加唯一ID，默认隐藏（只显示初始切片）
            display_style = 'block' if (initial_z_idx is not None and z_idx == initial_z_idx) else 'none'
            
            html += f"""
            <div id="slice-analysis-{z_idx}" class="slice-analysis-div" style="margin: 30px 0; padding: 20px; background-color: white; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: {display_style};">
                <h3 style="color: #3498db; border-bottom: 2px solid #3498db; padding-bottom: 10px;">
                    切片 Z={z_idx}
                </h3>
            """
            
            # 显示统计信息
            if stats:
                html += f"""
                <div style="background-color: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px;">
                    <strong>检测结果:</strong><br/>
                    面积: {stats.get('area', 0)} 像素<br/>
                    圆形度: {stats.get('circularity', 0):.3f}
                </div>
                """
            
            # 显示处理步骤图像
            html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin-top: 15px;">'
            
            if self.fixed_threshold is not None:
                binary_title = f'2. 二值化(固定阈值={self.fixed_threshold:g})'
            else:
                binary_title = f'2. 二值化({self.percentile:g}%分位)'

            step_names = {
                'original': '1. 原始切片',
                'binary': binary_title,
                'morphology': '3. 形态学操作',
                'connected_components': '4. 连通域标记',
                'merge_3d': '4.5 同切面3D合并',
                'area_filter': '5. 面积筛选',
                'distance': '6. 距离中心线筛选',
                'scoring': '7. 圆形度评分',
                'final': '8. 最终结果'
            }
            
            for key, title in step_names.items():
                if key in images:
                    html += f"""
                    <div style="text-align: center; padding: 10px; background-color: #fafafa; border-radius: 5px;">
                        <p style="margin: 5px 0; font-weight: bold; color: #555;">{title}</p>
                        <img src="data:image/png;base64,{images[key]}" 
                             style="max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 3px;"/>
                    </div>
                    """
            
            html += '</div></div>'
        
        html += '</div>'
        return html

    def _generate_experiment_meta_html(self, *, ended_at: datetime.datetime):
        """在页面底部展示本次实验参数/耗时/简介。"""
        started_at = self.experiment_started_at
        if isinstance(started_at, datetime.datetime):
            duration_s = (ended_at - started_at).total_seconds()
            started_str = started_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            duration_s = 0.0
            started_str = "N/A"
        ended_str = ended_at.strftime("%Y-%m-%d %H:%M:%S")

        # 仅保留可JSON化的参数
        safe_args = {}
        for k, v in (self.experiment_args or {}).items():
            try:
                json.dumps(v, ensure_ascii=False)
                safe_args[k] = v
            except Exception:
                safe_args[k] = str(v)

        # 补充关键内部参数，方便复现实验
        safe_args["_internal"] = {
            "percentile": self.percentile,
            "roi_size": self.roi_size,
            "kernel_size": list(self.kernel_size) if isinstance(self.kernel_size, tuple) else self.kernel_size,
            "2d_morph": "dilate(iter=1)",
            "3d_closing_iterations": self.closing_iters,
            "3d_erosion_iterations": self.erosion_iters,
        }

        intro = (self.experiment_intro or "").strip()
        intro_html = (
            intro.replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;")
        )

        args_pretty = json.dumps(safe_args, ensure_ascii=False, indent=2)

        return f"""
        <div style="margin-top: 40px; padding: 20px; background: #0b1220; color: #e5e7eb; border-top: 3px solid #2563eb; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
            <h2 style="margin: 0 0 10px; font-family: system-ui, -apple-system, sans-serif; color: #fff;">🧪 实验信息</h2>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 12px; font-family: system-ui, -apple-system, sans-serif;">
                <div><b>开始时间</b>: {started_str}</div>
                <div><b>结束时间</b>: {ended_str}</div>
                <div><b>耗时</b>: {duration_s:.2f} s</div>
                <div><b>输出前缀</b>: {self.output_name}</div>
            </div>

            <h3 style="margin: 14px 0 8px; font-family: system-ui, -apple-system, sans-serif; color:#fff;">简介（运行前命令行填写）</h3>
            <pre style="white-space: pre-wrap; background:#111827; padding:12px; border-radius:8px; margin:0 0 14px;">{intro_html if intro_html else "(空)"}</pre>

            <h3 style="margin: 14px 0 8px; font-family: system-ui, -apple-system, sans-serif; color:#fff;">参数（命令行 + 内部关键参数）</h3>
            <pre style="white-space: pre-wrap; background:#111827; padding:12px; border-radius:8px; margin:0;">{args_pretty}</pre>
        </div>
        """
    
    def run_full_pipeline(self, z_min=None, z_max=None, downsample_size=256,
                         iso_value=0.5, step_size=2, output_html=None,
                         show_endoscopy=False,
                         show_cross_sections=True, cross_section_interval=10,
                         auto_open=True, use_3d_analysis=False, use_flood_fill=True,
                         start_z=-100.0, start_idx=None,
                         navigation_line=False, nav_min_turn_radius_mm=12.0,
                         vtk_flythrough_mp4=None):
        """运行完整流程
        
        参数:
            use_3d_analysis: 是否使用3D分析
            use_flood_fill: True=充气法(3D连通性,推荐), False=传统传播法
            vtk_flythrough_mp4: 若为非空字符串，则在流程末尾尝试导出 PyVista 虚拟内镜 MP4（需安装 pyvista 与 imageio-ffmpeg）。
        """
        print("\n" + "🏥 "*30)
        print("DICOM气管3D重建流程")
        print("🏥 "*30)
        
        # 步骤1: 加载DICOM
        num_slices = self.step1_load_and_sort_dicom(z_min, z_max)
        if num_slices == 0:
            print("✗ 错误: 没有找到有效的DICOM切片")
            return False
        
        # 步骤2: 生成网格
        num_verts, num_faces = self.step3_generate_mesh(downsample_size, iso_value, step_size)
        if num_verts == 0:
            print("✗ 错误: 网格生成失败")
            return False
        
        # 步骤3: 可视化
        fig = self.step4_create_visualization(output_html, show_endoscopy,
                                              show_cross_sections, cross_section_interval,
                                              use_3d_analysis, use_flood_fill, start_z, start_idx,
                                              navigation_line, nav_min_turn_radius_mm)

        if vtk_flythrough_mp4:
            try:
                from virtual_endoscopy_pyvista import export_from_pipeline
                print("\n" + "=" * 60)
                print("PyVista 虚拟内镜（离屏 MP4）")
                print("=" * 60)
                okv = export_from_pipeline(self, vtk_flythrough_mp4)
                if okv and output_html:
                    print(f"  可与 HTML 对照: {output_html}")
            except Exception as e:
                print(f"⚠ 虚拟内镜 MP4 导出失败: {e}")
        
        # 总结
        print("\n" + "="*60)
        print("✓ 完成!")
        print("="*60)
        print(f"\n📊 重建统计:")
        print(f"  - DICOM切片: {num_slices}张")
        print(f"  - Z轴范围: {self.slices_data[0][0]:.1f}mm 到 {self.slices_data[-1][0]:.1f}mm")
        print(f"  - 网格顶点: {num_verts:,}个")
        print(f"  - 网格三角形: {num_faces:,}个")
        
        if output_html:
            print(f"\n💡 提示:")
            print(f"  - 在浏览器中打开: {output_html}")
            print(f"  - 鼠标拖动旋转、滚轮缩放")
            if show_endoscopy:
                print(f"  - 点击'播放'按钮: 虚拟内窥镜动画")
            
            # 自动打开浏览器
            if auto_open:
                import webbrowser
                webbrowser.open(f"file:///{os.path.abspath(output_html)}")
                print(f"\n🌐 已在浏览器中打开!")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="DICOM气管3D重建 + 中心线提取完整流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本使用(必填: Z范围 + 种子)
  python dicom_trachea_complete.py --dicom dicom文件夹 --z-min -119 --z-max -44 --start-z -100
  
  # 只重建主气管区域(Z=-119到-44mm)
  python dicom_trachea_complete.py --dicom dicom文件夹 --z-min -119 --z-max -44 --start-z -100
  
  # 添加虚拟内窥镜动画
  python dicom_trachea_complete.py --dicom dicom文件夹 --z-min -119 --z-max -44 --start-z -100 --endoscopy
  
  # 高质量重建
  python dicom_trachea_complete.py --dicom dicom文件夹 --z-min -119 --z-max -44 --start-z -100 --size 256 --step 1
  
  # 自定义输出
  python dicom_trachea_complete.py --dicom dicom文件夹 --z-min -119 --z-max -44 --start-z -100 --output my_trachea
        """
    )
    
    parser.add_argument('--dicom', required=True, help='DICOM文件夹路径')
    parser.add_argument('--output', '-o', default='trachea_reconstruction',
                       help='输出文件名前缀(默认: trachea_reconstruction)')
    parser.add_argument('--size', type=int, default=256,
                       help='降采样尺寸(默认: 256)')
    parser.add_argument('--iso', type=float, default=0.5,
                       help='Marching Cubes等值面(默认: 0.5)')
    parser.add_argument('--step', type=int, default=2,
                       help='Marching Cubes步长(默认: 2)')
    parser.add_argument('--percentile', type=float, default=36.5,
                       help='二值化分位阈值(默认: 36.5；越小越保守，越不易粘连；支持小数如35.5)')
    parser.add_argument('--fixed-threshold', type=float, default=None,
                       help='固定二值化阈值（窗位窗宽归一化后的[0,1]阈值）。设置后将覆盖 --percentile 的分位阈值策略。')
    parser.add_argument('--closing-iters', type=int, default=2,
                       help='3D closing 迭代次数(默认: 2)')
    parser.add_argument('--erosion-iters', type=int, default=1,
                       help='3D erosion 迭代次数(默认: 1；可设0表示不做erosion)')
    parser.add_argument('--intro', type=str, default="",
                       help='本次实验简介（用于记录思路/目的，写入HTML底部）')
    parser.add_argument('--z-min', type=float, required=True, help='Z轴最小值(mm)【必填】')
    parser.add_argument('--z-max', type=float, required=True, help='Z轴最大值(mm)【必填】')
    parser.add_argument('--no-cross-sections', action='store_true',
                       help='不显示横截面切片')
    parser.add_argument('--section-interval', type=int, default=15,
                       help='横截面间隔(默认: 15)')
    parser.add_argument('--endoscopy', action='store_true',
                       help='添加虚拟内窥镜动画')
    parser.add_argument('--open', action='store_true',
                       help='运行结束后自动打开浏览器(默认关闭)')
    parser.add_argument('--no-open', action='store_true',
                       help='不自动打开浏览器(兼容旧参数；默认本来就是不打开)')
    parser.add_argument('--use-3d-analysis', action='store_true',
                       help='使用3D分析(推荐启用)')
    parser.add_argument('--use-propagation', action='store_true',
                       help='使用传统传播法而非充气法(默认使用充气法)')
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument('--start-z', type=float, default=None,
                            help='3D分析起始Z坐标(mm)【必填二选一: --start-z 或 --start-idx】')
    seed_group.add_argument('--start-idx', type=int, default=None,
                            help='3D分析起始z_idx【必填二选一: --start-z 或 --start-idx】')

    parser.add_argument('--navigation-line', action='store_true',
                       help='从3D二值掩码提取插管导航线，并绘制到3D图层（仅用体素图+几何代价）')
    parser.add_argument('--nav-min-radius', type=float, default=12.0,
                       help='导航线最小转弯半径(mm)约束的目标值(默认: 12.0)')
    parser.add_argument(
        '--vtk-flythrough',
        nargs='?',
        default=None,
        const='',
        metavar='OUT.mp4',
        help='导出 PyVista 离屏虚拟内镜 MP4（内壁网格+射线锥前向）；不写路径则输出到 output/<前缀>_flythrough_<时间>.mp4。需: pip install pyvista imageio-ffmpeg',
    )
    
    args = parser.parse_args()
    
    # 根据DICOM文件夹自动调整参数(仅影响中心线提取和切片显示范围,不过滤DICOM加载)
    dicom_path = args.dicom.lower()
    if 'dicom文件夹2' in dicom_path or 'dicom2' in dicom_path:
        # dicom文件夹2: 切片间隔20
        if args.section_interval == 15:  # 如果是默认值,则调整
            args.section_interval = 20
        print(f"📋 检测到dicom文件夹2, 自动调整: 切片间隔={args.section_interval}")
    else:
        # dicom文件夹: 保持默认切片间隔15
        print(f"📋 检测到dicom文件夹, 切片间隔={args.section_interval}")
    
    # 创建输出目录
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}/")
    
    # 本次运行输出目录（按时间戳分组）
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, ts)
    if os.path.exists(run_dir):
        # 极少数情况下同秒多次启动，追加序号避免覆盖
        k = 2
        while os.path.exists(f"{run_dir}_{k}"):
            k += 1
        run_dir = f"{run_dir}_{k}"
    os.makedirs(run_dir, exist_ok=True)
    print(f"📦 本次输出目录: {run_dir}{os.sep}")

    # 输出文件路径
    output_html = os.path.join(run_dir, f"{args.output}_3d_{ts}.html")
    fly_mp4 = None
    if args.vtk_flythrough is not None:
        if args.vtk_flythrough == '':
            fly_mp4 = os.path.join(run_dir, f"{args.output}_flythrough_{ts}.mp4")
        else:
            fly_mp4 = args.vtk_flythrough
    
    # 创建流程对象
    pipeline = DicomTrachea3DPipeline(
        args.dicom,
        args.output,
        percentile=args.percentile,
        closing_iters=args.closing_iters,
        erosion_iters=args.erosion_iters,
        fixed_threshold=args.fixed_threshold,
    )
    pipeline.experiment_intro = args.intro or ""
    pipeline.experiment_args = vars(args).copy()
    pipeline.experiment_started_at = datetime.datetime.now()
    
    # 运行完整流程
    success = pipeline.run_full_pipeline(
        z_min=args.z_min,
        z_max=args.z_max,
        downsample_size=args.size,
        iso_value=args.iso,
        step_size=args.step,
        output_html=output_html,
        show_cross_sections=not args.no_cross_sections,
        cross_section_interval=args.section_interval,
        show_endoscopy=args.endoscopy,
        auto_open=(args.open and (not args.no_open)),
        use_3d_analysis=args.use_3d_analysis,
        use_flood_fill=not args.use_propagation,  # 默认使用充气法
        start_z=args.start_z,
        start_idx=args.start_idx,
        navigation_line=args.navigation_line,
        nav_min_turn_radius_mm=args.nav_min_radius,
        vtk_flythrough_mp4=fly_mp4,
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())
