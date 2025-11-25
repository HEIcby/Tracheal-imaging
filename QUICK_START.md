# 🎯 快速体验指南 - 三种方法看到效果

选择适合您的方式快速体验项目效果！

---

## 📋 目录
- [方法1: 一键批处理脚本（最简单）](#方法1-一键批处理脚本最简单)
- [方法2: Jupyter Notebook（最直观）](#方法2-jupyter-notebook最直观)
- [方法3: 分步手动运行（最灵活）](#方法3-分步手动运行最灵活)
- [常见问题](#常见问题)

---

## 方法1: 一键批处理脚本（最简单）⭐

**适合**: Windows用户，想要最快看到效果

### 完整流程（包含编译）
```bash
# 双击运行，或在命令行执行
.\run_full_demo.bat
```

这个脚本会自动：
1. ✅ 检查CMake和编译器
2. ✅ 编译Marching Cubes C++项目
3. ✅ 运行三维重建算法
4. ✅ 打开浏览器显示3D模型

**预计时间**: 5-10分钟（首次编译较慢）

### 快速查看（如果已编译）
```bash
# 如果已经有out.vtk文件
.\run_quick_demo.bat
```

**预计时间**: 30秒

---

## 方法2: Jupyter Notebook（最直观）⭐⭐⭐

**适合**: 想要交互式体验，逐步了解每个步骤

### 步骤1: 安装Jupyter环境
```bash
pip install jupyter numpy plotly matplotlib
```

### 步骤2: 启动快速演示Notebook
```bash
# 启动notebook
jupyter notebook Quick_Start_Demo.ipynb
```

### 步骤3: 按Shift+Enter运行每个单元格

**包含内容**:
- 📊 可视化原始医学图像切片
- 🔧 运行Marching Cubes算法
- 🎨 交互式3D重建模型展示
- 💡 详细的步骤说明

**预计时间**: 10分钟

### 其他推荐Notebook:

#### 🏥 CT肝脏分割完整流程
```bash
jupyter notebook jupyter_lab/ai_liver/ct_3d_reconstruction_clean.ipynb
```
- 从CT图像到三维重建的完整流程
- 使用ITK进行交互式3D可视化

#### 🎨 高级3D渲染
```bash
jupyter notebook jupyter_lab/reconstruction/plotly_vtk_render.ipynb
```
- 读取VTK文件并渲染
- 自定义视觉效果

#### 🔬 图像预处理演示
```bash
jupyter notebook jupyter_lab/preprocess/preprocess.ipynb
```
- 中值滤波、USM锐化效果对比
- 对比度增强前后对比

---

## 方法3: 分步手动运行（最灵活）

**适合**: 想要深入了解每个步骤，或者需要自定义参数

### 3.1 编译Marching Cubes

```bash
cd marching_cubes_c
mkdir build
cd build

# CMake配置
cmake ..

# 编译（Windows MSVC）
cmake --build . --config Release

# 或者编译（MinGW）
cmake --build .
```

### 3.2 运行三维重建

```bash
# 回到marching_cubes_c目录
cd ..

# Windows MSVC编译输出
.\build\Release\marching_cubes_c.exe --input case_00000_x.npy --iso 0.5 --vtk out.vtk

# 或MinGW编译输出
.\build\marching_cubes_c.exe --input case_00000_x.npy --iso 0.5 --vtk out.vtk
```

**参数说明**:
- `--input`: 输入NPY文件路径
- `--iso`: 等值面阈值（可以尝试0.3-0.7之间的值）
- `--vtk`: 输出VTK文件路径

### 3.3 可视化结果

#### 方法A: Python脚本
```bash
# 安装依赖
pip install numpy plotly

# 运行可视化
python quick_demo_viewer.py marching_cubes_c\out.vtk
```

#### 方法B: 直接打开HTML
生成的 `out_visualization.html` 可以直接用浏览器打开

#### 方法C: 使用ParaView（专业可视化）
1. 下载安装 [ParaView](https://www.paraview.org/download/)
2. 打开 `out.vtk` 文件
3. 点击 "Apply" 查看3D模型

---

## 🎨 AI分割演示

### 查看预训练模型效果

```bash
cd ai_segmentation/code2

# 安装依赖
pip install -r requirements.txt

# 运行推理（如果有测试数据）
python test/inference.py ^
    --model_path models/ChaosCT_0.9770885167.pth ^
    --input_dir test_images ^
    --output_dir results
```

### Jupyter演示

```bash
# 细胞分割
jupyter notebook jupyter_lab/ai_cell/batch_2dunet.ipynb

# 肝脏分割
jupyter notebook jupyter_lab/ai_liver/liver_segmentation_dpu.ipynb
```

---

## 📊 数据预处理演示

### HLS仿真（需要Vivado HLS）

```bash
cd data_preprocessing/hls_testbench
# 在Vivado HLS中打开项目运行C仿真
```

### Python预处理演示

```bash
jupyter notebook jupyter_lab/preprocess/preprocess.ipynb
```

---

## 🔧 常见问题

### Q1: 运行批处理脚本时提示"找不到CMake"
**解决**: 
1. 下载安装 [CMake](https://cmake.org/download/)
2. 安装时选择"Add CMake to PATH"
3. 重启命令行窗口

### Q2: 提示"找不到编译器"
**解决**:
- **方案A**: 安装 [Visual Studio 2019/2022](https://visualstudio.microsoft.com/) 
  - 安装时勾选"使用C++的桌面开发"
- **方案B**: 使用Developer Command Prompt运行脚本
- **方案C**: 安装 [MinGW-w64](https://www.mingw-w64.org/)

### Q3: Jupyter Notebook无法打开
**解决**:
```bash
# 确保安装了Jupyter
pip install jupyter

# 如果是权限问题
pip install --user jupyter

# 启动
jupyter notebook
```

### Q4: Python导入错误
**解决**:
```bash
# 安装所有依赖
pip install numpy plotly matplotlib vtk pillow itk itkwidgets

# 如果网络慢，使用国内镜像
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy plotly matplotlib
```

### Q5: 3D模型显示为空白
**检查**:
1. VTK文件是否成功生成（检查文件大小>0）
2. 等值参数是否合适（尝试0.3-0.7之间的值）
3. 浏览器是否支持WebGL（尝试Chrome或Firefox）

### Q6: 想要使用自己的医学图像
**步骤**:
1. 将DICOM/NIfTI转换为NPY格式
   ```bash
   jupyter notebook jupyter_lab/preprocess/transform_nii_npy.ipynb
   ```
2. 使用转换后的NPY文件运行Marching Cubes
3. 可能需要调整等值参数

---

## 📁 输出文件说明

运行演示后会生成以下文件：

| 文件 | 说明 | 大小估计 |
|------|------|----------|
| `out.vtk` | 三维网格文件（VTK格式） | 1-10 MB |
| `out_visualization.html` | 交互式HTML可视化 | 2-5 MB |
| `demo_output.vtk` | 演示输出（如果指定） | 1-10 MB |

---

## 🎯 推荐学习路径

### 初学者
1. ✅ 运行 `run_full_demo.bat` 看到第一个效果
2. ✅ 打开 `Quick_Start_Demo.ipynb` 了解流程
3. ✅ 尝试修改等值参数，观察效果变化

### 进阶用户
1. ✅ 研究 Marching Cubes 算法实现
2. ✅ 尝试 AI分割模块
3. ✅ 探索数据预处理流程
4. ✅ 使用自己的医学图像数据

### 开发者
1. ✅ 阅读源代码 `marching_cubes_c/src/`
2. ✅ 理解HLS实现 `marching_cubes_hls/`
3. ✅ 修改和优化算法
4. ✅ 部署到KV260硬件平台

---

## 📚 更多资源

- 📖 [完整项目文档](README.md)
- 🔧 [AI分割模块文档](ai_segmentation/code2/README.md)
- 🎨 [数据预处理文档](data_preprocessing/README.md)
- 🏗️ [Marching Cubes C++文档](marching_cubes_c/README.md)
- ⚡ [Marching Cubes HLS文档](marching_cubes_hls/README.md)

---

## 💬 获得帮助

遇到问题？
1. 查看本文档的[常见问题](#常见问题)章节
2. 阅读相关模块的README文档
3. 检查终端输出的错误信息

---

**祝您体验愉快！** 🎉
