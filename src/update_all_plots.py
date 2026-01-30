import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置统一的学术绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# 参考图中的配色方案
COLORS = {
    'red': '#f59790',
    'blue': '#9eaad1',
    'yellow': '#f5dbb6',
    'purple': '#dacfe5',
    'cyan': '#cde2e8',
    'light_blue': '#c8d4e9'
}

# 统一指标配色
METRIC_COLORS = {
    'Recall': COLORS['red'],
    'Precision': COLORS['blue'],
    'F1-score': COLORS['yellow']
}

def fix_algorithm_performance():
    df = pd.read_csv(r'C:\Users\Echo\Desktop\modbus-detection\results\tables\phase3_algorithm_comparison.csv')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = ['test_recall', 'test_precision', 'test_f1']
    labels = ['Recall', 'Precision', 'F1-score']
    x = np.arange(len(df['algorithm']))
    width = 0.25

    for i, metric in enumerate(metrics):
        ax1.bar(x + i*width, df[metric], width, label=labels[i], color=METRIC_COLORS[labels[i]], edgecolor='gray', linewidth=0.5, alpha=0.9)

    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('(a) Detection Performance Metrics', fontsize=14)
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(df['algorithm'], rotation=15)
    ax1.set_ylim(0, 1.1)
    ax1.legend(loc='lower right')

    # 推理时间图使用参考图中的多种颜色
    bar_colors = [COLORS['blue'], COLORS['purple'], COLORS['cyan'], COLORS['red'], COLORS['yellow']]
    bars = ax2.bar(df['algorithm'], df['predict_time_per_sample_ms'], color=bar_colors[:len(df)], edgecolor='gray', linewidth=0.5, alpha=0.9)
    ax2.set_ylabel('Inference Time (ms/sample)', fontsize=12)
    ax2.set_title('(b) Computational Efficiency', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['algorithm'], rotation=15)

    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height, f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(r'C:\Users\Echo\Desktop\modbus-detection\results\figures\algorithm_performance_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

def fix_ablation_comparison():
    df = pd.read_csv(r'C:\Users\Echo\Desktop\modbus-detection\results\tables\phase3_ablation_results.csv')
    group_map = {
        '组1-仅协议层': 'Protocol Only',
        '组2-仅时序层': 'Temporal Only',
        '组3-仅DCS业务逻辑': 'Operation Only',
        '组4-协议层+时序层': 'Protocol + Temporal',
        '组5-全部特征': 'All Features'
    }
    df['group_en'] = df['group'].map(group_map)
    
    plt.figure(figsize=(10, 6))
    metrics = ['test_recall', 'test_precision', 'test_f1']
    labels = ['Recall', 'Precision', 'F1-score']
    x = np.arange(len(df['group_en']))
    width = 0.25

    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df[metric], width, label=labels[i], color=METRIC_COLORS[labels[i]], edgecolor='gray', linewidth=0.5, alpha=0.9)

    plt.ylabel('Score', fontsize=12)
    plt.title('Ablation Study: Impact of Feature Categories', fontsize=14)
    plt.xticks(x + width, df['group_en'], rotation=15)
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Echo\Desktop\modbus-detection\results\figures\ablation_comparison_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

def fix_feature_importance():
    df = pd.read_csv(r'C:\Users\Echo\Desktop\modbus-detection\results\tables\phase2_feature_importance.csv').head(10)
    plt.figure(figsize=(10, 6))
    # 使用参考图中的多种配色循环
    bar_colors = [COLORS['blue'], COLORS['red'], COLORS['yellow'], COLORS['purple'], COLORS['cyan'], 
                  COLORS['light_blue'], COLORS['blue'], COLORS['red'], COLORS['yellow'], COLORS['purple']]
    
    bars = plt.barh(df['feature'][::-1], df['importance'][::-1], color=bar_colors[::-1], edgecolor='gray', linewidth=0.5, alpha=0.9)
    
    plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
    plt.ylabel('Feature Name', fontsize=12, fontweight='bold')
    plt.title('Top 10 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    
    # 在条形图末端添加数值标注
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                 va='center', fontsize=10, fontweight='bold', color='gray')
        
    plt.tight_layout()
    plt.savefig(r'C:\Users\Echo\Desktop\modbus-detection\results\figures\feature_importance_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

def fix_two_layer_comparison():
    data = {
        'Method': ['Pure ML (RF)', 'Pure Rule', 'Two-Layer (Rule+RF)'],
        'Accuracy': [0.9907, 0.9471, 0.9907],
        'Precision': [0.9496, 1.0000, 0.9496],
        'Recall': [0.8813, 0.0594, 0.8813],
        'F1-score': [0.9142, 0.1121, 0.9142]
    }
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    metrics = ['Recall', 'Precision', 'F1-score']
    x = np.arange(len(df['Method']))
    width = 0.25

    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df[metric], width, label=metric, color=METRIC_COLORS[metric], edgecolor='gray', linewidth=0.5, alpha=0.9)

    plt.ylabel('Score', fontsize=12)
    plt.title('Comparison of Detection Architectures', fontsize=14)
    plt.xticks(x + width, df['Method'])
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Echo\Desktop\modbus-detection\results\figures\two_layer_comparison_v2.png', dpi=300, bbox_inches='tight')
    plt.close()

def fix_confusion_matrix():
    # 模拟混淆矩阵数据 (基于实验结果)
    # Normal: 18596 (TN), 52 (FP)
    # Attack: 132 (FN), 980 (TP)
    cm = np.array([[18596, 52], [132, 980]])
    plt.figure(figsize=(8, 6))
    # 使用参考图中的浅蓝色调作为 cmap
    cmap = sns.light_palette(COLORS['blue'], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                xticklabels=['Normal', 'Attack'], yticklabels=['Normal', 'Attack'])
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title('Confusion Matrix (XGBoost)', fontsize=14)
    plt.tight_layout()
    plt.savefig(r'C:\Users\Echo\Desktop\modbus-detection\results\figures\confusion_matrix_test.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    fix_algorithm_performance()
    fix_ablation_comparison()
    fix_feature_importance()
    fix_two_layer_comparison()
    fix_confusion_matrix()
    print("All figures updated with the new coordinated color palette.")
