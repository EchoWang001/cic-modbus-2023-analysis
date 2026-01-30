import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置绘图风格
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# 读取数据
df = pd.read_csv(r'C:\Users\Echo\Desktop\modbus-detection\results\tables\phase3_algorithm_comparison.csv')

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- 左图：性能指标对比 ---
metrics = ['test_recall', 'test_precision', 'test_f1']
labels = ['Recall', 'Precision', 'F1-score']
x = np.arange(len(df['algorithm']))
width = 0.25

for i, metric in enumerate(metrics):
    ax1.bar(x + i*width, df[metric], width, label=labels[i], alpha=0.8)

ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('(a) Detection Performance Metrics', fontsize=14, fontweight='bold')
ax1.set_xticks(x + width)
ax1.set_xticklabels(df['algorithm'], rotation=15)
ax1.set_ylim(0, 1.1)
ax1.legend(loc='lower right')

# --- 右图：推理时间对比 ---
# 使用对数坐标或直接标注，因为差异巨大
bars = ax2.bar(df['algorithm'], df['predict_time_per_sample_ms'], color=sns.color_palette("muted"), alpha=0.8)
ax2.set_ylabel('Inference Time (ms/sample)', fontsize=12)
ax2.set_title('(b) Computational Efficiency', fontsize=14, fontweight='bold')
ax2.set_xticklabels(df['algorithm'], rotation=15)

# 在柱状图上方添加数值标注
for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()

# 保存图片
output_path = r'C:\Users\Echo\Desktop\modbus-detection\results\figures\algorithm_performance_v2.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"New figure saved to: {output_path}")
