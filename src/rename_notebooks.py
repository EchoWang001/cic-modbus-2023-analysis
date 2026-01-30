import os
from pathlib import Path

notebook_dir = Path(r'C:\Users\Echo\Desktop\modbus-detection\notebooks')

rename_map = {
    "01_环境测试.ipynb": "01_Environment_Test.ipynb",
    "03_数据采样_阶段1_探索.ipynb": "03_Data_Sampling_Phase1_Exploration.ipynb",
    "04_数据采样_阶段2_快速测试.ipynb": "04_Data_Sampling_Phase2_Quick_Test.ipynb",
    "05_标签诊断检查.ipynb": "05_Label_Diagnostic_Check.ipynb",
    "06_数据集质量排查.ipynb": "06_Dataset_Quality_Audit.ipynb",
    "07_Phase1_数据准备.ipynb": "07_Phase1_Data_Preparation.ipynb",
    "08_Phase2_基础实验.ipynb": "08_Phase2_Baseline_Experiments.ipynb",
    "09_Phase3_核心验证.ipynb": "09_Phase3_Core_Validation.ipynb",
    "10_Phase3.5_补充实验.ipynb": "10_Phase3.5_Supplementary_Experiments.ipynb",
    "11_Phase3.5_E9_误判分析.ipynb": "11_Phase3.5_Error_Analysis.ipynb"
}

for old_name, new_name in rename_map.items():
    old_path = notebook_dir / old_name
    new_path = notebook_dir / new_name
    if old_path.exists():
        os.rename(old_path, new_path)
        print(f"Renamed: {old_name} -> {new_name}")
    else:
        print(f"Not found: {old_name}")
