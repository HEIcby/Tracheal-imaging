# 快速演示指南 - 快速看到效果

## 🎯 方案1：三维重建演示（最快，5分钟）

### 步骤1：安装依赖
```bash
pip install numpy plotly vtk pyvista
```

### 步骤2：运行Marching Cubes三维重建
```bash
cd marching_cubes_c
mkdir build
cd build
cmake ..
cmake --build . --config Release
cd ..
.\build\Release\marching_cubes_c.exe --input case_00000_x.npy --iso 0.5 --vtk demo_output.vtk
```

### 步骤3：可视化结果
打开Jupyter Notebook查看3D效果：
```bash
jupyter notebook
# 打开 jupyter_lab/reconstruction/plotly_vtk_render.ipynb
# 修改第一个cell中的文件名为 'demo_output.vtk'
```

---

## 🎯 方案2：交互式Jupyter演示（最直观，10分钟）

### 适合快速体验整个流程，有可视化界面

1. **安装Jupyter环境**
```bash
pip install jupyter numpy pillow itk itkwidgets plotly vtk
```

2. **启动Jupyter Lab**
```bash
cd jupyter_lab
jupyter lab
```

3. **选择演示Notebook**（按推荐顺序）：

#### 🥇 推荐1：CT三维重建完整演示
- 打开：`ai_liver/ct_3d_reconstruction_clean.ipynb`
- 功能：展示从CT图像到三维重建的完整流程
- 效果：交互式3D可视化

#### 🥈 推荐2：3D可视化渲染
- 打开：`reconstruction/plotly_vtk_render.ipynb`
- 功能：使用Plotly渲染VTK文件
- 效果：可旋转的3D模型，医学风格渲染

#### 🥉 推荐3：数据预处理演示
- 打开：`preprocess/preprocess.ipynb`
- 功能：展示图像滤波、锐化、增强效果
- 效果：对比预处理前后的图像

---

## 🎯 方案3：AI分割快速测试（需要GPU，15分钟）

### 查看预训练模型效果

1. **安装PyTorch环境**
```bash
cd ai_segmentation/code2
pip install -r requirements.txt
```

2. **准备测试数据**
```bash
# 如果有自己的医学图像，放在 test_images/ 目录
# 或使用项目提供的数据
```

3. **运行推理**
```bash
# Windows PowerShell:
python test/inference.py --model_path models/ChaosCT_0.9770885167.pth --input_dir test_images --output_dir results
```

4. **查看结果**
- 分割结果保存在 `results/` 目录
- 可以使用图像查看器对比原图和分割mask

---

## 📊 快速可视化脚本

如果你想最快看到效果，我可以为您创建一个一键演示脚本：

### Python可视化脚本（推荐使用）
