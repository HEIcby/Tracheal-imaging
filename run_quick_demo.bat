@echo off
REM ========================================
REM 医学图像处理流水线 - 快速演示脚本
REM ========================================

echo.
echo ========================================
echo   医学图像处理流水线快速演示
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.7+
    pause
    exit /b 1
)

echo [1/4] 检查并安装依赖包...
pip install numpy plotly -q
if errorlevel 1 (
    echo [警告] 某些包安装失败，但继续尝试运行
)

echo [2/4] 检查是否存在VTK文件...
if exist "marching_cubes_c\out.vtk" (
    echo [找到] marching_cubes_c\out.vtk
    set VTK_FILE=marching_cubes_c\out.vtk
) else (
    echo [未找到] 需要先生成VTK文件
    echo.
    echo 请先运行以下命令生成三维模型:
    echo   cd marching_cubes_c
    echo   mkdir build
    echo   cd build
    echo   cmake ..
    echo   cmake --build . --config Release
    echo   cd ..
    echo   .\build\Release\marching_cubes_c.exe --input case_00000_x.npy --iso 0.5 --vtk out.vtk
    echo.
    echo 或者查看 quick_demo.md 了解其他演示方法
    pause
    exit /b 1
)

echo [3/4] 启动3D可视化程序...
python quick_demo_viewer.py %VTK_FILE%

echo.
echo [4/4] 演示完成！
echo.
echo 更多演示方法请查看: quick_demo.md
echo.
pause
