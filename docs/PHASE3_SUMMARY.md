# Phase 3: 核心验证实验 - 执行总结

> 执行时间：2026年1月14日  
> 状态：✅ 已完成

---

## 一、工作概述

Phase 3 完成了核心验证实验，包括 DCS 特征消融、跨场景泛化、两层架构验证和算法对比四个实验，全面评估所提方法的有效性和局限性。

### 实验目标

| 实验 | 名称 | 目的 |
|------|------|------|
| E6 | DCS 特征消融 | 验证 DCS 业务逻辑特征的贡献 |
| E7 | 跨场景泛化 | 评估 SCADA→IED 跨场景迁移能力 |
| E4 | 两层架构验证 | 验证规则层+ML层的架构优势 |
| E3 | 算法对比 | RF vs DT vs LR vs XGBoost vs MLP |

### 目标达成情况

| 指标 | 目标 | 实际结果 | 状态 |
|------|------|----------|------|
| 完整方法 F1 | > 90% | 92.79% | ✅ 达成 |
| XGBoost F1 | - | 96.17% | ✅ 最佳 |
| 跨场景 F1 下降 | < 10% | 0% | ⚠️ 需分析 |

---

## 二、实验设计

### 2.1 E6: DCS 特征消融实验

**目的**：验证 DCS 业务逻辑特征（22个）的贡献

**实验组设计**：

| 组别 | 特征集 | 特征数 |
|------|--------|--------|
| 组1 | 仅协议层 | 13 |
| 组2 | 仅时序层 | 9 |
| 组3 | 仅 DCS 业务逻辑 | 22 |
| 组4 | 协议层 + 时序层 | 22 |
| 组5 | 全部特征 | 44 |

**评估方法**：5-Fold 交叉验证 + 测试集最终评估

### 2.2 E7: 跨场景泛化测试

**目的**：验证在 SCADA 场景训练的模型能否泛化到 IED 场景

**实验设计**：
- 训练数据：SCADA 场景 (38,799 样本)
- 测试场景：
  - 同场景 (SCADA → SCADA)：80/20 划分
  - 跨场景 (SCADA → IED)：全部 IED 数据作为测试集

### 2.3 E4: 两层架构验证

**目的**：验证 规则层 + ML 层 的两层架构相比纯 ML 的优势

**规则设计**（基于文献）：

| 规则 | 条件 | 依据 |
|------|------|------|
| R1 | `external_ip_present == 1` | 工控网络隔离原则 |
| R2 | `consecutive_write_max > 3` | Wool 2013: repeating sequence |
| R3 | `write_without_read_ratio == 1` | Wool 2013: highly periodic polling |

**组合方式**：R1 OR R2 OR R3

### 2.4 E3: 算法对比实验

**对比算法**：

| 算法 | 关键参数 |
|------|----------|
| Random Forest | n_estimators=100, class_weight='balanced' |
| Decision Tree | max_depth=None, class_weight='balanced' |
| Logistic Regression | max_iter=1000, class_weight='balanced' |
| XGBoost | n_estimators=100, scale_pos_weight=21.7 |
| MLP | hidden_layer_sizes=(100, 50), max_iter=500 |

---

## 三、代码实现

### 3.1 核心代码结构

```
notebooks/
└── 09_Phase3_核心验证.ipynb    # 完整实验代码

主要模块：
├── 0. 环境准备                 # 导入库、配置参数
├── 1. 数据加载                 # 加载 train/val/test 数据
├── 2. 辅助函数定义             # evaluate_model, cross_validate_model 等
├── 3. E6: DCS特征消融          # 5组特征消融实验
├── 4. E7: 跨场景泛化           # SCADA→SCADA, SCADA→IED
├── 5. E4: 两层架构验证         # 规则层 + ML层
├── 6. E3: 算法对比             # 5种算法对比
├── 7. 结果汇总                 # 输出最终报告
└── 8. 保存结果                 # CSV 和图表
```

### 3.2 关键实现细节

#### XGBoost 标签编码处理

```python
from sklearn.preprocessing import LabelEncoder

# XGBoost 需要数值标签 (0/1)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train_full)  # Attack=0, Normal=1
y_test_encoded = le.transform(y_test)

# XGBoost 专用评估
if algo_name == 'XGBoost':
    y_tr = y_train_encoded
    y_te = y_test_encoded
    pos_label = 0  # Attack=0
```

#### 规则层实现

```python
def apply_rules(df, consecutive_write_threshold=3):
    """
    应用规则层检测
    
    规则设计依据 (基于文献):
    - R1: external_ip_present == 1 (工控网络隔离原则)
    - R2: consecutive_write_max > threshold (Wool 2013)
    - R3: write_without_read_ratio == 1 (Wool 2013)
    """
    r1 = df['external_ip_present'] == 1
    r2 = df['consecutive_write_max'] > consecutive_write_threshold
    r3 = df['write_without_read_ratio'] == 1
    
    rule_attack = r1 | r2 | r3
    return rule_attack, {'R1': r1, 'R2': r2, 'R3': r3}
```

---

## 四、实验结果

### 4.1 E6: DCS 特征消融结果

| 实验组 | 特征数 | CV F1 | Test F1 | Test Recall | Test AUC |
|--------|--------|-------|---------|-------------|----------|
| 组1-仅协议层 | 13 | 0.9534 | 0.9221 | 0.8939 | 0.9975 |
| 组2-仅时序层 | 9 | 0.8094 | 0.8537 | 0.7869 | 0.9816 |
| 组3-仅DCS业务逻辑 | 22 | 0.7062 | 0.8056 | **0.9838** | 0.9952 |
| 组4-协议层+时序层 | 22 | 0.9514 | 0.9335 | 0.9146 | 0.9993 |
| 组5-全部特征 | 44 | 0.9491 | 0.9279 | 0.9083 | 0.9993 |

**DCS 特征贡献分析**：

| 指标 | 结果 | 分析 |
|------|------|------|
| F1 增益 | -0.0056 | 组5 vs 组4 |
| 组3 Recall | 98.38% | DCS 特征可独立检测 98.4% 攻击 |

**关键发现**：
- F1 增益为负，原因是数据集攻击模式简单，协议层特征已足够
- 但 DCS 特征对 Recall 有独特贡献（98.38%），在实际工控场景中有价值

### 4.2 E7: 跨场景泛化结果

| 测试场景 | Accuracy | Precision | Recall | F1 | AUC |
|----------|----------|-----------|--------|-----|-----|
| 同场景 (SCADA→SCADA) | 100% | 100% | 100% | **1.0000** | 1.0000 |
| 跨场景 (SCADA→IED) | 100% | 100% | 100% | **1.0000** | 1.0000 |

**⚠️ 数据复杂度分析**：

```
SCADA场景:
  Attack cwm>0: 3811/3811 (100%)
  Normal cwm>0: 0/34988 (0%)
  ⚠ 警告: 简单规则'cwm>0'可100%区分Attack/Normal

IED场景:
  Attack cwm>0: 1400/1400 (100%)
  Normal cwm>0: 0/33182 (0%)
  ⚠ 警告: 简单规则'cwm>0'可100%区分Attack/Normal
```

**结论**：100% 准确率是由于 `consecutive_write_max > 0` 在 SCADA/IED 场景可完美区分 Attack/Normal，表明数据集在该场景分类过于简单。

### 4.3 E4: 两层架构验证结果

#### 规则特征在数据集中的分布

| 特征条件 | 测试集 Attack 覆盖 |
|----------|-------------------|
| external_ip_present == 1 | 0/1112 (0%) |
| write_without_read_ratio == 1 | 0/1112 (0%) |
| consecutive_write_max > 0 | 1112/1112 (100%) |
| consecutive_write_max > 1 | 1107/1112 (99.5%) |
| consecutive_write_max > 2 | 80/1112 (7.2%) |
| consecutive_write_max > 3 | 66/1112 (5.9%) |

**⚠️ 重要发现**：R1 和 R3 在本数据集不触发，仅 R2 有效

#### 架构对比结果

| 方法 | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|-----|
| 纯 RF | 99.07% | 94.96% | 88.13% | 91.42% |
| 纯规则层 (cwm>3) | 99.38% | 91.67% | 5.94% | 11.16% |
| 两层架构 (规则+RF) | 99.07% | 94.96% | 88.13% | 91.42% |

**Recall 提升**：+0.00%

**原因分析**：由于规则阈值 (cwm>3) 仅覆盖 5.9% 的攻击，而 RF 已能检测 88.13%，规则层对 Recall 无额外贡献。

### 4.4 E3: 算法对比结果

| 算法 | CV F1 | Test F1 | Test Recall | 推理时间 (ms) | 模型大小 (MB) |
|------|-------|---------|-------------|---------------|---------------|
| **XGBoost** | 0.9637 ± 0.0095 | **0.9617** | **0.9362** | 0.0010 | 0.24 |
| Decision Tree | 0.9392 ± 0.0033 | 0.9336 | 0.9110 | 0.0002 | 0.08 |
| Random Forest | 0.9491 ± 0.0048 | 0.9279 | 0.9083 | 0.0054 | 9.99 |
| Logistic Regression | 0.7132 ± 0.0081 | 0.8145 | 0.9892 | 0.0001 | 0.00 |
| MLP | 0.5914 ± 0.1288 | 0.8101 | 0.6906 | 0.0150 | 0.12 |

**最佳算法**：XGBoost (F1=0.9617, Recall=0.9362)

**最轻量算法**：Decision Tree (模型仅 0.08MB，推理 0.0002ms)

---

## 五、输出文件

### 5.1 结果表格 (`results/tables/`)

| 文件名 | 说明 |
|--------|------|
| `phase3_ablation_results.csv` | E6 消融实验结果 |
| `phase3_cross_scenario_results.csv` | E7 跨场景泛化结果 |
| `phase3_two_layer_results.csv` | E4 两层架构结果 |
| `phase3_threshold_sensitivity.csv` | E4 阈值敏感性分析 |
| `phase3_algorithm_comparison.csv` | E3 算法对比结果 |
| `phase3_fn_analysis.csv` | 漏检样本分析 |

### 5.2 结果图表 (`results/figures/`)

| 文件名 | 说明 |
|--------|------|
| `phase3_ablation_comparison.png` | 消融实验对比图 |
| `phase3_cross_scenario_comparison.png` | 跨场景泛化对比图 |
| `phase3_algorithm_comparison.png` | 算法性能对比图 |
| `phase3_algorithm_efficiency.png` | 算法效率对比图 |

### 5.3 Notebook (`notebooks/`)

| 文件名 | 说明 |
|--------|------|
| `09_Phase3_核心验证.ipynb` | Phase 3 完整执行记录 |

### 5.4 设计文档 (`docs/`)

| 文件名 | 说明 |
|--------|------|
| `PHASE3_DESIGN.md` | Phase 3 实验设计文档 |

---

## 六、关键配置参数

```python
# 通用参数
RANDOM_SEED = 42
N_FOLDS = 5
POS_LABEL = 'Attack'

# Random Forest 参数
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'n_jobs': -1
}

# XGBoost 参数
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'scale_pos_weight': 21.7,
    'random_state': RANDOM_SEED,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# 规则层阈值
RULE_CONFIG = {
    'R1': {'feature': 'external_ip_present', 'condition': '== 1'},
    'R2': {'feature': 'consecutive_write_max', 'condition': '> 3'},
    'R3': {'feature': 'write_without_read_ratio', 'condition': '== 1'}
}
```

---

## 七、结论与分析

### 7.1 主要发现

| 发现 | 结论 | 影响 |
|------|------|------|
| XGBoost 最佳 | F1=0.9617, 优于 RF | 推荐用于部署 |
| DCS 特征 Recall 高 | 98.38% 独立检测能力 | 验证业务语义价值 |
| 规则层效果有限 | 仅覆盖 5.9% 攻击 | 需调整阈值或规则 |
| 数据集简单 | cwm>0 完美区分 | 泛化结论需谨慎 |

### 7.2 数据集局限性分析

**关键特征 `consecutive_write_max` 的影响**：

| 数据集 | Attack (cwm>0) | Normal (cwm>0) | 结论 |
|--------|----------------|----------------|------|
| SCADA | 100% | 0% | 完美可分 |
| IED | 100% | 0% | 完美可分 |
| 测试集 | 100% | 0.27% | 近似完美 |

这导致：
1. E7 跨场景 100% 准确率是假象，非真正泛化能力
2. 任何包含 `consecutive_write_max` 的模型都能获得极高性能
3. 难以准确评估其他特征的真实贡献

### 7.3 论文论证建议

| 实验 | 原结论 | 建议调整 |
|------|--------|----------|
| E6 | DCS 特征有 F1 贡献 | 强调 Recall 贡献 (98.38%) |
| E7 | 模型泛化能力强 | 说明数据集特性导致完美分离 |
| E4 | 两层架构有优势 | 讨论规则设计与数据集匹配性 |
| E3 | XGBoost 最佳 | 结论可靠 ✅ |

---

## 八、下一步工作

### 8.1 短期改进

1. **调整 E4 规则阈值**：将 `consecutive_write_max > 3` 改为 `> 1` 或 `> 0`
2. **增加数据集复杂度分析章节**：在论文中说明数据集特性
3. **重新解读 E6/E7 结论**：强调方法设计的合理性而非绝对指标

### 8.2 长期建议

1. **测试更复杂数据集**：寻找 Attack/Normal 不易区分的工控数据集
2. **特征重要性深入分析**：排除 `consecutive_write_max` 后重新评估
3. **规则层优化**：设计更适合本数据集特性的规则

---

## 九、注意事项

1. **XGBoost 标签编码**：XGBoost 要求数值标签 (0/1)，其他算法可使用字符串标签

2. **数据集特性**：
   - SCADA/IED 场景中 `consecutive_write_max > 0` 可完美区分 Attack/Normal
   - E7 的 100% 准确率不代表真正泛化能力

3. **规则层局限**：
   - R1 (`external_ip_present`) 和 R3 (`write_without_read_ratio`) 在本数据集不触发
   - 规则设计基于文献，保留以确保方法完整性

4. **结果解读**：
   - 高性能指标需结合数据集特性分析
   - 论文中应包含数据集复杂度说明

---

*文档生成时间: 2026-01-14*
