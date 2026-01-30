@echo off
echo ========================================
echo 启动Jupyter Notebook
echo ========================================

REM 激活conda环境
call conda activate modbus-detection

REM 启动Jupyter
jupyter notebook

pause