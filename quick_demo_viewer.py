"""
快速演示脚本 - 三维医学图像可视化
用于快速查看 Marching Cubes 三维重建效果
"""

import numpy as np
import plotly.graph_objects as go
import sys
import os

def read_vtk_file(vtk_path):
    """读取VTK文件并提取顶点和三角形"""
    print(f"正在读取VTK文件: {vtk_path}")
    
    try:
        import vtk
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(vtk_path)
        reader.Update()
        polydata = reader.GetOutput()
        
        # 提取顶点
        points = polydata.GetPoints()
        vertices = np.array([points.GetPoint(i) for i in range(points.GetNumberOfPoints())])
        
        # 提取三角形
        triangles = []
        for i in range(polydata.GetNumberOfCells()):
            cell = polydata.GetCell(i)
            if cell.GetNumberOfPoints() == 3:
                triangles.append([cell.GetPointId(j) for j in range(3)])
        triangles = np.array(triangles)
        
        print(f"✓ 成功读取 {len(vertices)} 个顶点, {len(triangles)} 个三角形")
        return vertices, triangles
        
    except ImportError:
        print("警告: vtk库未安装，尝试手动解析VTK文件...")
        return read_vtk_manual(vtk_path)

def read_vtk_manual(vtk_path):
    """手动解析VTK Legacy格式文件（简化版）"""
    vertices = []
    triangles = []
    
    with open(vtk_path, 'r') as f:
        lines = f.readlines()
        
        # 查找POINTS部分
        i = 0
        while i < len(lines):
            if 'POINTS' in lines[i]:
                num_points = int(lines[i].split()[1])
                i += 1
                for _ in range(num_points):
                    coords = list(map(float, lines[i].split()))
                    vertices.append(coords[:3])
                    i += 1
                break
            i += 1
        
        # 查找POLYGONS部分
        while i < len(lines):
            if 'POLYGONS' in lines[i]:
                num_polys = int(lines[i].split()[1])
                i += 1
                for _ in range(num_polys):
                    poly_data = list(map(int, lines[i].split()))
                    if poly_data[0] == 3:  # 三角形
                        triangles.append(poly_data[1:4])
                    i += 1
                break
            i += 1
    
    vertices = np.array(vertices)
    triangles = np.array(triangles)
    print(f"✓ 手动解析成功: {len(vertices)} 个顶点, {len(triangles)} 个三角形")
    return vertices, triangles

def visualize_3d_mesh(vertices, triangles, title="医学图像三维重建"):
    """使用Plotly创建交互式3D可视化"""
    
    # 翻转Z轴（医学图像常用）
    vertices[:, 2] = -vertices[:, 2]
    
    # 计算顶点深度用于颜色映射
    z_values = vertices[:, 2]
    
    # 创建3D mesh
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=triangles[:, 0],
            j=triangles[:, 1],
            k=triangles[:, 2],
            intensity=z_values,
            colorscale='Viridis',  # 可选: 'Plasma', 'Inferno', 'Hot', 'Bone'
            showscale=True,
            lighting=dict(
                ambient=0.4,
                diffuse=0.8,
                specular=0.5,
                roughness=0.3,
                fresnel=0.2
            ),
            lightposition=dict(
                x=100,
                y=200,
                z=0
            ),
            flatshading=False,
            opacity=1.0,
            name='3D模型'
        )
    ])
    
    # 设置布局
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center',
            font=dict(size=20, color='white')
        ),
        scene=dict(
            xaxis=dict(
                title='X轴',
                backgroundcolor="rgb(20, 24, 54)",
                gridcolor="rgb(50, 50, 50)",
                showbackground=True
            ),
            yaxis=dict(
                title='Y轴',
                backgroundcolor="rgb(20, 24, 54)",
                gridcolor="rgb(50, 50, 50)",
                showbackground=True
            ),
            zaxis=dict(
                title='Z轴',
                backgroundcolor="rgb(20, 24, 54)",
                gridcolor="rgb(50, 50, 50)",
                showbackground=True
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        paper_bgcolor="rgb(10, 10, 10)",
        plot_bgcolor="rgb(10, 10, 10)",
        font=dict(color='white'),
        width=1200,
        height=900,
        showlegend=True
    )
    
    return fig

def main():
    """主函数"""
    print("=" * 60)
    print("  医学图像三维重建快速演示")
    print("=" * 60)
    
    # 检查输入参数
    if len(sys.argv) > 1:
        vtk_path = sys.argv[1]
    else:
        # 默认使用项目中的输出文件
        vtk_path = 'marching_cubes_c/out.vtk'
        if not os.path.exists(vtk_path):
            print(f"\n❌ 未找到VTK文件: {vtk_path}")
            print("\n使用方法:")
            print("  python quick_demo_viewer.py [vtk_file_path]")
            print("\n示例:")
            print("  python quick_demo_viewer.py marching_cubes_c/out.vtk")
            print("\n请先运行Marching Cubes生成VTK文件:")
            print("  cd marching_cubes_c/build")
            print("  ./marching_cubes_c --input ../case_00000_x.npy --iso 0.5 --vtk ../out.vtk")
            return
    
    if not os.path.exists(vtk_path):
        print(f"❌ 文件不存在: {vtk_path}")
        return
    
    # 读取VTK文件
    vertices, triangles = read_vtk_file(vtk_path)
    
    if len(vertices) == 0 or len(triangles) == 0:
        print("❌ 未能从VTK文件中提取有效数据")
        return
    
    # 显示统计信息
    print("\n📊 模型统计:")
    print(f"  • 顶点数量: {len(vertices):,}")
    print(f"  • 三角形数量: {len(triangles):,}")
    print(f"  • X范围: [{vertices[:, 0].min():.2f}, {vertices[:, 0].max():.2f}]")
    print(f"  • Y范围: [{vertices[:, 1].min():.2f}, {vertices[:, 1].max():.2f}]")
    print(f"  • Z范围: [{vertices[:, 2].min():.2f}, {vertices[:, 2].max():.2f}]")
    
    # 创建可视化
    print("\n🎨 正在生成3D可视化...")
    fig = visualize_3d_mesh(vertices, triangles, 
                           title=f"医学图像三维重建 - {os.path.basename(vtk_path)}")
    
    # 显示
    print("✓ 可视化准备完成！")
    print("\n💡 提示:")
    print("  • 使用鼠标拖动旋转模型")
    print("  • 使用滚轮缩放")
    print("  • 双击重置视图")
    print("\n🌐 正在打开浏览器...")
    
    fig.show()
    
    # 保存HTML文件
    html_path = vtk_path.replace('.vtk', '_visualization.html')
    fig.write_html(html_path)
    print(f"\n💾 可视化已保存到: {html_path}")
    print("   可以直接用浏览器打开此HTML文件查看")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 用户中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
