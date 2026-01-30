@echo off
echo ========================================
echo 启动完整数据采样（过夜运行）
echo ========================================
echo.
echo 预计耗时: 4-6小时
echo 日志文件: results\logs\sampling_log.txt
echo.
echo 按任意键开始...
pause

REM 激活conda环境
call conda activate modbus-detection

REM 运行采样脚本
python src\full_sampling.py

echo.
echo ========================================
echo 采样完成！
echo ========================================
pause