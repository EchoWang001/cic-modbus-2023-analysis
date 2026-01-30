import nbformat
import os
from pathlib import Path

notebook_dir = Path(r'C:\Users\Echo\Desktop\modbus-detection\notebooks')

# Common translation map for ICS/Modbus project
translation_map = {
    "阶段": "Phase",
    "实验": "Experiment",
    "数据准备": "Data Preparation",
    "特征提取": "Feature Extraction",
    "模型训练": "Model Training",
    "评估": "Evaluation",
    "结果": "Results",
    "分析": "Analysis",
    "混淆矩阵": "Confusion Matrix",
    "准确率": "Accuracy",
    "精确率": "Precision",
    "召回率": "Recall",
    "特征重要性": "Feature Importance",
    "消融实验": "Ablation Study",
    "核心验证": "Core Validation",
    "补充实验": "Supplementary Experiments",
    "误判分析": "Error Analysis",
    "基础实验": "Baseline Experiments",
    "环境测试": "Environment Test",
    "数据采样": "Data Sampling",
    "探索": "Exploration",
    "快速测试": "Quick Test",
    "标签诊断": "Label Diagnostics",
    "质量排查": "Quality Audit",
    "协议层": "Protocol Layer",
    "时序层": "Temporal Layer",
    "业务逻辑": "Business Logic",
    "全部特征": "All Features",
    "二分类": "Binary Classification",
    "多分类": "Multi-class Classification",
    "两层架构": "Two-layer Architecture",
    "规则层": "Rule Layer",
    "跨场景": "Cross-scenario",
    "泛化能力": "Generalization",
    "正常": "Normal",
    "攻击": "Attack",
    "样本": "Samples",
    "时间窗口": "Time Window"
}

def translate_notebook(file_path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    print(f"Translating: {file_path.name}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            # Simple replacement for markdown cells
            for cn, en in translation_map.items():
                cell.source = cell.source.replace(cn, en)
        elif cell.cell_type == 'code':
            # Translate comments in code cells
            lines = cell.source.split('\n')
            new_lines = []
            for line in lines:
                if '#' in line:
                    parts = line.split('#', 1)
                    comment = parts[1]
                    for cn, en in translation_map.items():
                        comment = comment.replace(cn, en)
                    new_lines.append(f"{parts[0]}# {comment}")
                else:
                    new_lines.append(line)
            cell.source = '\n'.join(new_lines)

    with open(file_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Finished: {file_path.name}")

# List of notebooks to translate
target_notebooks = [
    "07_Phase1_Data_Preparation.ipynb",
    "08_Phase2_Baseline_Experiments.ipynb",
    "09_Phase3_Core_Validation.ipynb",
    "10_Phase3.5_Supplementary_Experiments.ipynb",
    "11_Phase3.5_Error_Analysis.ipynb"
]

for nb_name in target_notebooks:
    translate_notebook(notebook_dir / nb_name)
