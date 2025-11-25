@echo off
REM ========================================
REM 完整演示流程 - 从数据到可视化
REM ========================================

echo.
echo ========================================
echo   完整演示流程（包含编译和重建）
echo ========================================
echo.

REM 检查CMake
cmake --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到CMake，请先安装CMake
    echo 下载地址: https://cmake.org/download/
    pause
    exit /b 1
)

REM 检查编译器
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo [警告] 未检测到MSVC编译器
    echo 请安装Visual Studio 2019/2022或使用Developer Command Prompt
    echo.
    where g++.exe >nul 2>&1
    if errorlevel 1 (
        echo [错误] 也未检测到MinGW g++编译器
        pause
        exit /b 1
    ) else (
        echo [信息] 将使用MinGW编译器
        set USE_MINGW=1
    )
)

echo [1/5] 进入Marching Cubes目录...
cd marching_cubes_c
if not exist "case_00000_x.npy" (
    echo [错误] 测试数据文件不存在: case_00000_x.npy
    cd ..
    pause
    exit /b 1
)

echo [2/5] 创建构建目录...
if not exist "build" mkdir build
cd build

echo [3/5] 配置CMake项目...
if defined USE_MINGW (
    cmake -G "MinGW Makefiles" ..
) else (
    cmake ..
)
if errorlevel 1 (
    echo [错误] CMake配置失败
    cd ..\..
    pause
    exit /b 1
)

echo [4/5] 编译项目（这可能需要几分钟）...
cmake --build . --config Release
if errorlevel 1 (
    echo [错误] 编译失败
    cd ..\..
    pause
    exit /b 1
)

echo [5/5] 运行Marching Cubes算法...
cd ..
if exist "build\Release\marching_cubes_c.exe" (
    echo [使用] build\Release\marching_cubes_c.exe
    .\build\Release\marching_cubes_c.exe --input case_00000_x.npy --iso 0.5 --vtk out.vtk
) else if exist "build\marching_cubes_c.exe" (
    echo [使用] build\marching_cubes_c.exe
    .\build\marching_cubes_c.exe --input case_00000_x.npy --iso 0.5 --vtk out.vtk
) else (
    echo [错误] 未找到编译后的可执行文件
    cd ..
    pause
    exit /b 1
)

if errorlevel 1 (
    echo [错误] Marching Cubes运行失败
    cd ..
    pause
    exit /b 1
)

echo.
echo ========================================
echo   VTK文件生成成功！
echo ========================================
echo.

cd ..

echo 正在启动3D可视化...
echo.
pip install numpy plotly -q
python quick_demo_viewer.py marching_cubes_c\out.vtk

echo.
echo ========================================
echo   演示完成！
echo ========================================
echo.
echo 生成的文件:
echo   • marching_cubes_c\out.vtk - 三维网格文件
echo   • marching_cubes_c\out_visualization.html - 可视化HTML
echo.
pause
