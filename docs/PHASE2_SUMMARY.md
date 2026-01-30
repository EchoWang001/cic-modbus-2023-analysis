# Phase 2: 基础实验 - 执行总结

> 执行时间：2026年1月14日  
> 状态：✅ 已完成

---

## 一、工作概述

Phase 2 完成了 Random Forest 基线模型的训练与评估，验证了所提出的特征工程方案在 Modbus 异常检测任务上的有效性。

### 实验目标

1. **E1: 二分类异常检测** - 验证 RF 模型的基本性能
2. **E2: 5-Fold 交叉验证** - 增强结果可靠性，评估模型稳定性

### 预期目标

| 指标 | 目标值 | 达成状态 |
|------|--------|---------|
| Accuracy | > 95% | ✅ 达成 (99.07%) |
| F1-Score | > 90% | ✅ 达成 (91.42%) |
| 推理时间 | < 10ms/样本 | ✅ 达成 (0.0043ms) |

---

## 二、实验设计

### 2.1 数据划分

采用 Phase 1 准备的数据集，按 PCAP 文件划分以防止数据泄漏：

| 数据集 | 样本数 | Attack 样本 | Attack 比例 |
|--------|--------|-------------|-------------|
| 训练集 | 83,078 | 3,346 | 4.03% |
| 验证集 | 15,371 | 760 | 4.94% |
| 测试集 | 19,760 | 1,112 | 5.63% |
| **总计** | **118,209** | **5,218** | **4.41%** |

### 2.2 模型配置

根据设计文档 5.3 节，Random Forest 参数配置如下：

```python
RF_PARAMS = {
    'n_estimators': 100,        # 树的数量
    'max_depth': None,          # 不限制深度，让树充分生长
    'min_samples_split': 2,     # 最小分裂样本数
    'min_samples_leaf': 1,      # 叶节点最小样本数
    'max_features': 'sqrt',     # 特征采样：sqrt(n_features)
    'class_weight': 'balanced', # 处理类别不平衡
    'random_state': 42,         # 随机种子
    'n_jobs': -1,               # 并行处理
    'oob_score': True           # 记录 OOB 分数
}
```

### 2.3 评估指标

| 指标 | 说明 | 重要性 |
|------|------|--------|
| Accuracy | 整体分类准确率 | 基础指标 |
| Precision | 预测为攻击中实际是攻击的比例 | 避免误报 |
| Recall | 实际攻击中被正确检测的比例 | **工控安全核心指标** |
| F1-Score | Precision 和 Recall 的调和平均 | 综合评价 |
| AUC-ROC | ROC 曲线下面积 | 排序能力 |
| AUC-PR | PR 曲线下面积 | **不平衡数据关键指标** |

### 2.4 交叉验证方案

- 方法：5-Fold Stratified Cross-Validation
- 数据：仅在训练集上进行
- 目的：评估模型稳定性，避免单次划分的随机性影响

---

## 三、代码实现

### 3.1 核心代码结构

```
notebooks/
└── 08_Phase2_基础实验.ipynb    # 完整实验代码

主要模块：
├── 0. 环境准备                 # 导入库、配置路径
├── 1. 数据加载                 # 加载 train/val/test 数据
├── 2. RF 模型训练 (E1)         # 主模型训练
├── 3. 测试集评估               # 计算各项指标
├── 4. 5-Fold 交叉验证 (E2)     # 稳定性验证
├── 5. 可视化                   # 混淆矩阵、ROC/PR 曲线等
├── 6. 特征重要性分析           # 初步分析 Top-10 特征
├── 7. 模型保存                 # 保存模型和元数据
└── 8. 结果汇总                 # 输出最终报告
```

### 3.2 关键代码片段

#### 数据加载与特征分离

```python
from config import DATA_SPLITS
from feature_extractor import get_feature_names

# 加载数据
df_train = pd.read_parquet(DATA_SPLITS / 'train.parquet')
df_test = pd.read_parquet(DATA_SPLITS / 'test.parquet')

# 获取 44 个特征名
feature_names = get_feature_names()

# 分离特征和标签
X_train = df_train[feature_names].values
y_train = df_train['label'].values  # 字符串标签: 'Normal' / 'Attack'
```

#### 模型训练

```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(**RF_PARAMS)
rf_model.fit(X_train, y_train)

print(f"OOB Score: {rf_model.oob_score_:.4f}")
```

#### 评估指标计算

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# 预测
y_pred = rf_model.predict(X_test)

# 预测概率 (注意: classes_ 按字母序排列, Attack=0, Normal=1)
y_pred_proba = rf_model.predict_proba(X_test)[:, 0]  # Attack 类概率

# 计算指标
POS_LABEL = 'Attack'
metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred, pos_label=POS_LABEL),
    'recall': recall_score(y_test, y_pred, pos_label=POS_LABEL),
    'f1': f1_score(y_test, y_pred, pos_label=POS_LABEL),
    'auc_roc': roc_auc_score((y_test == POS_LABEL).astype(int), y_pred_proba),
    'auc_pr': average_precision_score((y_test == POS_LABEL).astype(int), y_pred_proba)
}
```

#### 5-Fold 交叉验证

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
    
    fold_model = RandomForestClassifier(**RF_PARAMS)
    fold_model.fit(X_fold_train, y_fold_train)
    
    y_fold_pred = fold_model.predict(X_fold_val)
    y_fold_proba = fold_model.predict_proba(X_fold_val)[:, 0]
    
    # 计算并记录每折指标...
```

### 3.3 重要实现细节

#### 标签类型处理

由于 Parquet 文件中的标签是字符串类型 (`'Normal'`, `'Attack'`)，需要注意：

1. **sklearn 指标函数**：使用 `pos_label='Attack'` 参数
2. **AUC 计算**：需要将标签转换为二进制 `(y == 'Attack').astype(int)`
3. **概率索引**：`classes_` 按字母顺序排列，`'Attack'` 在索引 0

#### 混淆矩阵标签顺序

```python
# 明确指定标签顺序，确保 ravel() 正确映射到 tn, fp, fn, tp
cm = confusion_matrix(y_test, y_pred, labels=['Normal', 'Attack'])
tn, fp, fn, tp = cm.ravel()
```

---

## 四、实验结果

### 4.1 测试集性能 (E1)

| 指标 | 结果 |
|------|------|
| **Accuracy** | 99.07% |
| **Precision** | 94.96% |
| **Recall** | 88.13% |
| **F1-Score** | 91.42% |
| **AUC-ROC** | 0.9991 |
| **AUC-PR** | 0.9846 |

### 4.2 混淆矩阵

```
                  Predicted
                  Normal    Attack
Actual  Normal    18,596      52      (误报率 FPR: 0.28%)
        Attack       132     980      (漏报率 FNR: 11.87%)
```

| 指标 | 数值 | 说明 |
|------|------|------|
| True Negative (TN) | 18,596 | 正确识别的正常流量 |
| False Positive (FP) | 52 | **误报** - 将正常识别为攻击 |
| False Negative (FN) | 132 | **漏报** - 未检测到的攻击 |
| True Positive (TP) | 980 | 正确检测的攻击 |

### 4.3 5-Fold 交叉验证 (E2)

| 指标 | 均值 | 标准差 | 最小值 | 最大值 |
|------|------|--------|--------|--------|
| Accuracy | 99.57% | 0.02% | 99.54% | 99.60% |
| Precision | 96.02% | 0.38% | 95.58% | 96.73% |
| Recall | 93.07% | 0.61% | 92.39% | 93.87% |
| F1-Score | 94.52% | 0.32% | 94.14% | 94.94% |
| AUC-ROC | 99.95% | 0.00% | 99.95% | 99.96% |
| AUC-PR | 98.90% | 0.11% | 98.71% | 99.05% |

**论文报告格式**：
- Accuracy: 99.57% ± 0.02%
- Precision: 96.02% ± 0.38%
- Recall: 93.07% ± 0.61%
- F1-Score: 94.52% ± 0.32%

### 4.4 轻量级指标

| 指标 | 结果 | 目标 | 状态 |
|------|------|------|------|
| 训练时间 | 3.68 秒 | - | 快速 |
| 推理时间 | 0.0043 ms/样本 | < 10ms | ✅ 远超目标 |
| 模型大小 | 9.93 MB | - | 可接受 |
| OOB Score | 0.9960 | - | 泛化能力强 |

### 4.5 Top-10 重要特征

| 排名 | 特征名 | 重要性 | 特征类型 |
|------|--------|--------|----------|
| 1 | write_without_read_ratio | ~0.15 | DCS业务逻辑 |
| 2 | fc_write_ratio | ~0.12 | 协议层 |
| 3 | consecutive_write_max | ~0.10 | DCS业务逻辑 |
| 4 | operation_sequence_entropy | ~0.08 | DCS业务逻辑 |
| 5 | consecutive_write_mean | ~0.07 | DCS业务逻辑 |
| 6 | write_burst_count | ~0.06 | DCS业务逻辑 |
| 7 | fc_read_ratio | ~0.05 | 协议层 |
| 8 | read_write_alternation | ~0.04 | DCS业务逻辑 |
| 9 | packet_rate | ~0.03 | 时序层 |
| 10 | interval_std | ~0.03 | 时序层 |

**发现**：Top-10 中有 6 个是 DCS 业务逻辑特征，验证了核心创新点的有效性。

---

## 五、输出文件

### 5.1 模型文件 (`models/`)

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `rf_baseline.pkl` | 9.93 MB | 训练好的 RF 模型 |
| `rf_baseline_metadata.json` | <1 KB | 模型元信息（参数、指标等） |

### 5.2 结果图表 (`results/figures/`)

| 文件名 | 说明 |
|--------|------|
| `confusion_matrix_test.png` | 测试集混淆矩阵热力图 |
| `roc_curve_test.png` | ROC 曲线 |
| `pr_curve_test.png` | Precision-Recall 曲线 |
| `probability_distribution.png` | 预测概率分布 |
| `cv_results.png` | 交叉验证结果对比 |
| `feature_importance_top20.png` | Top-20 特征重要性 |

### 5.3 结果表格 (`results/tables/`)

| 文件名 | 说明 |
|--------|------|
| `phase2_test_metrics.csv` | 测试集评估指标 |
| `phase2_cv_results.csv` | 交叉验证详细结果 |
| `phase2_feature_importance.csv` | 特征重要性排名 |

### 5.4 Notebook (`notebooks/`)

| 文件名 | 说明 |
|--------|------|
| `08_Phase2_基础实验.ipynb` | Phase 2 完整执行记录 |

---

## 六、关键配置参数

```python
# Random Forest 参数
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'oob_score': True
}

# 交叉验证
N_FOLDS = 5

# 正类标签
POS_LABEL = 'Attack'

# 随机种子
RANDOM_SEED = 42
```

---

## 七、模型加载示例

### 加载模型进行预测

```python
import pickle
import numpy as np
from feature_extractor import get_feature_names

# 加载模型
with open('models/rf_baseline.pkl', 'rb') as f:
    model = pickle.load(f)

# 获取特征名
feature_names = get_feature_names()

# 假设 X_new 是新的特征数据 (shape: [n_samples, 44])
y_pred = model.predict(X_new)
y_proba = model.predict_proba(X_new)[:, 0]  # Attack 类概率

print(f"预测结果: {y_pred}")
print(f"Attack 概率: {y_proba}")
```

### 加载模型元信息

```python
import json

with open('models/rf_baseline_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"训练时间: {metadata['train_time']}")
print(f"测试集 F1: {metadata['test_metrics']['f1']}")
```

---

## 八、结论与分析

### 8.1 主要发现

1. **模型性能优秀**：
   - 所有核心指标 (Accuracy, F1, AUC) 均超过 90%
   - AUC-ROC 达到 0.9991，AUC-PR 达到 0.9846

2. **交叉验证稳定**：
   - 各指标标准差极小 (< 1%)
   - 表明模型泛化能力强，无过拟合

3. **轻量级要求满足**：
   - 推理时间 0.0043ms/样本，远低于 10ms 目标
   - 适合工控实时检测场景

4. **特征工程有效**：
   - DCS 业务逻辑特征在 Top-10 中占据主导
   - 验证了基于业务语义的特征设计的有效性

### 8.2 待改进方向

1. **召回率**：88.13% 的 Recall 意味着约 12% 的攻击被漏检
   - 工控安全场景中漏报的代价可能更高
   - Phase 3 可探索阈值调整或其他算法

2. **误报控制**：虽然 FPR 仅 0.28%，但在高频流量下仍可能产生较多告警
   - 可结合规则层进行二次过滤

---

## 九、下一步工作

**Phase 3: 对比实验**

| 实验 | 内容 | 目的 |
|------|------|------|
| E3 | 算法对比 | RF vs DT vs LR vs XGBoost vs MLP |
| E4 | 架构对比 | 纯规则 vs 纯ML vs 两层架构 |
| E6 | DCS特征消融 | 验证 DCS 业务逻辑特征的贡献 |

---

## 十、注意事项

1. **标签类型**：数据中的标签是字符串类型 (`'Normal'`, `'Attack'`)，使用 sklearn 函数时需注意 `pos_label` 参数

2. **概率索引**：`predict_proba()` 返回的概率列顺序与 `classes_` 一致，按字母序排列：
   - 索引 0 → `'Attack'`
   - 索引 1 → `'Normal'`

3. **类别不平衡**：Normal:Attack ≈ 21.7:1，已通过 `class_weight='balanced'` 处理

4. **模型版本**：当前模型基于 `max_depth=None`（完全生长），后续可尝试限制深度进行优化

---

*文档生成时间: 2026-01-14*
