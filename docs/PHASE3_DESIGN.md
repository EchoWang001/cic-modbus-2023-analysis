# Phase 3: 核心验证实验 - 设计文档

> 创建时间：2026年1月14日  
> 状态：✅ 设计完成，待实施  
> 依赖：Phase 1（数据准备）、Phase 2（基础实验）

---

## 一、Phase 3 目标

Phase 3 是"核心验证实验"阶段，主要目标是：

1. **验证核心创新点**：通过消融实验量化 DCS 业务逻辑特征的贡献
2. **验证泛化能力**：测试模型在不同 DCS 场景下的表现
3. **优化检测性能**：探索两层架构能否提升 Recall
4. **算法对比**：证明轻量级模型的适用性

---

## 二、Phase 2 关键结果回顾

### 2.1 RF 基线模型性能

| 指标 | 测试集结果 | 5-Fold CV (均值±标准差) |
|------|-----------|------------------------|
| Accuracy | 99.07% | 99.57% ± 0.02% |
| Precision | 94.96% | 96.02% ± 0.38% |
| **Recall** | **88.13%** | 93.07% ± 0.61% |
| F1-Score | 91.42% | 94.52% ± 0.32% |
| AUC-ROC | 0.9991 | 99.95% ± 0.00% |

### 2.2 混淆矩阵

```
                  Predicted
                  Normal    Attack
Actual  Normal    18,596      52      (FPR: 0.28%)
        Attack       132     980      (FNR: 11.87%)
```

**关键发现**：
- **132个FN**：约12%的攻击被漏检，是 Phase 3 E4 实验的优化目标
- **Top-10特征中6个是DCS特征**：初步验证了核心创新点

### 2.3 Top-10 重要特征

| 排名 | 特征名 | 特征类型 |
|------|--------|----------|
| 1 | write_without_read_ratio | **DCS业务逻辑** |
| 2 | fc_write_ratio | 协议层 |
| 3 | consecutive_write_max | **DCS业务逻辑** |
| 4 | operation_sequence_entropy | **DCS业务逻辑** |
| 5 | consecutive_write_mean | **DCS业务逻辑** |
| 6 | write_burst_count | **DCS业务逻辑** |
| 7 | fc_read_ratio | 协议层 |
| 8 | read_write_alternation | **DCS业务逻辑** |
| 9 | packet_rate | 时序层 |
| 10 | interval_std | 时序层 |

---

## 三、实验列表与执行顺序

### 3.1 实验概览

| 顺序 | 编号 | 实验名称 | 目的 | 优先级 |
|------|------|---------|------|--------|
| 1 | **E6** | DCS特征消融 | 验证核心创新点 | ⭐⭐⭐ 必做 |
| 2 | **E7** | 跨场景泛化 | 验证泛化能力 | ⭐⭐⭐ 必做 |
| 3 | **E4** | 两层架构验证 | 提升Recall | ⭐⭐⭐ 必做 |
| 4 | **E3** | 算法对比 | 证明轻量级优势 | ⭐⭐ 推荐 |

### 3.2 执行顺序说明

采用 **E6 → E7 → E4 → E3** 顺序，理由：

1. **逻辑递进**：先验证"特征有效性"，再验证"泛化能力"，最后"优化架构"
2. **E7为E4提供信息**：跨场景结果可能揭示FN样本的本质特征
3. **与论文结构匹配**：先展示方法有效性，再展示优化方案

---

## 四、E6: DCS业务逻辑特征消融实验

### 4.1 实验目的

量化验证 22 个 DCS 业务逻辑特征对检测性能的贡献，这是本研究的**核心创新点验证**。

### 4.2 实验组设计

| 实验组 | 使用特征 | 特征数 | 说明 |
|--------|---------|--------|------|
| 组1 | 仅协议层 | 13 | Baseline：传统Modbus协议特征 |
| 组2 | 仅时序层 | 9 | Baseline：通用流量统计特征 |
| 组3 | 仅DCS业务逻辑 | 22 | 验证DCS特化特征的独立能力 |
| 组4 | 协议层 + 时序层 | 22 | 无业务逻辑的对照组 |
| 组5 | 全部特征 | 44 | 完整方法（Phase 2基线） |

### 4.3 特征分组详情

#### 协议层特征 (13个)

```python
PROTOCOL_FEATURES = [
    'fc_distribution_entropy',  # 功能码分布熵
    'fc_diversity',             # 功能码种类数
    'fc_read_ratio',            # 读操作比例
    'fc_write_ratio',           # 写操作比例
    'txid_mean',                # 事务ID均值
    'txid_std',                 # 事务ID标准差
    'txid_unique_ratio',        # 唯一TxID比例
    'unit_id_diversity',        # 单元ID多样性
    'packet_size_mean',         # 平均包大小
    'packet_size_std',          # 包大小标准差
    'address_range',            # 访问地址范围
    'value_mean',               # 写入值均值
    'value_std',                # 写入值标准差
]
```

#### 时序层特征 (9个)

```python
TEMPORAL_FEATURES = [
    'packet_count',     # 窗口内包数量
    'packet_rate',      # 包速率 (packets/sec)
    'interval_mean',    # 包间隔均值
    'interval_std',     # 包间隔标准差
    'interval_min',     # 最小包间隔
    'interval_max',     # 最大包间隔
    'burst_count',      # 突发次数
    'burst_intensity',  # 突发强度
    'session_count',    # 会话数量
]
```

#### DCS业务逻辑特征 (22个)

```python
DCS_FEATURES = [
    # 设备角色特征 (6个)
    'scada_src_ratio',          # SCADA作为源的比例
    'scada_dst_ratio',          # SCADA作为目标的比例
    'ied_src_ratio',            # IED作为源的比例
    'ied_dst_ratio',            # IED作为目标的比例
    'device_role_entropy',      # 设备角色分布熵
    'is_typical_scada_to_ied',  # 是否为典型SCADA→IED模式
    
    # 通信拓扑特征 (5个)
    'src_ip_count',             # 唯一源IP数量
    'dst_ip_count',             # 唯一目标IP数量
    'comm_pair_count',          # 通信对数量
    'dominant_pair_ratio',      # 主要通信对占比
    'multi_target_ratio',       # 多目标操作比例
    
    # 操作模式特征 (6个) - 核心
    'consecutive_write_max',    # 最大连续写入次数
    'consecutive_write_mean',   # 平均连续写入长度
    'write_burst_count',        # 写入突发次数
    'read_write_alternation',   # 读写交替频率
    'operation_sequence_entropy', # 操作序列熵
    'write_without_read_ratio', # 无读取的写入比例
    
    # 异常指标特征 (5个)
    'external_ip_present',      # 是否存在外部IP
    'external_ip_packet_ratio', # 外部IP包占比
    'unknown_ip_count',         # 未知IP数量
    'abnormal_fc_ratio',        # 异常功能码比例
    'address_range_exceeded',   # 地址范围异常
]
```

### 4.4 实验步骤

1. 使用 `get_feature_groups()` 获取特征分组
2. 对每个实验组，使用对应的特征子集：
   - **训练集 5-fold CV**：在训练集上进行 5-fold Stratified CV，报告均值±标准差（评估模型稳定性）
   - **测试集最终评估**：使用全部训练集训练模型，在测试集上评估最终性能
3. 评估指标：Accuracy, Precision, Recall, F1, AUC-ROC
4. 绘制对比图表

> **评估方式统一说明**：所有实验（E6/E7/E3）均采用"训练集 5-fold CV + 测试集最终评估"的方式，确保结果的可比性和稳定性。

### 4.5 预期结论

- **组5（全部）性能最优**：验证完整方法的有效性
- **组3（仅DCS）有独特贡献**：证明DCS特征对Attack检测的重要性
- **组4 vs 组5的差异**：量化DCS业务逻辑特征的增益

### 4.6 输出文件

| 文件 | 说明 |
|------|------|
| `results/tables/phase3_ablation_results.csv` | 消融实验结果表 |
| `results/figures/ablation_comparison.png` | 性能对比图 |
| `results/figures/dcs_feature_contribution.png` | DCS特征贡献可视化 |

---

## 五、E7: 跨场景泛化测试

### 5.1 实验目的

验证在 SCADA 场景训练的模型能否泛化到 IED 场景，证明方法的通用性。

### 5.2 数据分布

| 场景 | 窗口数 | Normal | Attack | Attack比例 |
|------|--------|--------|--------|-----------|
| Compromised-SCADA | 38,799 | 33,681 | 5,118 | 13.2% |
| Compromised-IED | 34,582 | 34,483 | 99 | 0.29% |

### 5.3 实验设计

> ⚠️ **重要说明**：E7 实验需要**重新训练**仅使用 SCADA 数据的模型，而非使用 Phase 2 的模型（Phase 2 模型是在全部数据上训练的）。这样才能公平对比同场景与跨场景的性能差异。

#### 主实验：正向泛化

| 设置 | 训练数据 | 测试数据 | 模型 | 说明 |
|------|---------|---------|------|------|
| **同场景基准** | **仅SCADA** | SCADA测试集 | 新训练 | 同场景性能上限 |
| **跨场景** | **仅SCADA** | **IED全部** | 同上 | **核心验证** |

#### 补充实验：反向泛化（可选）

| 设置 | 训练数据 | 测试数据 | 模型 | 说明 |
|------|---------|---------|------|------|
| 反向泛化 | 仅IED | SCADA全部 | 新训练 | 补充分析（样本量限制） |

**注意**：
- IED场景仅99个Attack样本，反向泛化结果仅供参考
- Phase 2 的结果可作为"全数据训练"的参照，但不作为E7的"同场景基准"

### 5.4 实验步骤

#### 正向泛化（主实验）
1. 从 `features_15s_filtered.parquet` 筛选 Compromised-SCADA 场景数据（38,799窗口）
2. 按 70/15/15 划分训练/验证/测试集（按PCAP文件划分，防止数据泄漏）
3. **重新训练 RF 模型**（仅使用SCADA训练集）
4. 在 SCADA 训练集上进行 5-fold CV，报告均值±标准差
5. 在 SCADA 测试集上评估（同场景基准）
6. 在 IED 全部数据上评估（跨场景测试）
7. 对比性能差异，分析原因

#### 反向泛化（补充实验）
1. 从完整数据集中筛选 Compromised-IED 场景数据（34,582窗口，仅99 Attack）
2. 按 70/15/15 划分训练/验证/测试集
3. **重新训练 RF 模型**（仅使用IED训练集）
4. 在 IED 训练集上进行 5-fold CV
5. 在 SCADA 全部数据上评估
6. 记录结果，**明确标注样本量限制**

### 5.5 预期结果

```
┌─────────────────────────────────────────────────────────────────────┐
│  预期结果:                                                          │
│  - 跨场景性能下降 10-20% 是合理的                                    │
│  - 若下降过多 → 分析原因（攻击类型差异？设备行为差异？）              │
│  - 论文价值: 证明方法可推广到类似工控场景                            │
├─────────────────────────────────────────────────────────────────────┤
│  IED场景特殊性:                                                     │
│  - Attack样本仅99个，可能影响评估稳定性                              │
│  - 攻击类型可能与SCADA场景不同                                       │
│  - 需在论文中说明此局限性                                            │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.6 输出文件

| 文件 | 说明 |
|------|------|
| `results/tables/phase3_cross_scenario_results.csv` | 跨场景泛化结果 |
| `results/figures/cross_scenario_comparison.png` | 泛化性能对比图 |

---

## 六、E4: 两层架构验证

### 6.1 实验目的

验证基于 DCS 领域知识的规则层能否与 ML 层互补，提升 Recall（减少漏检）。

### 6.2 理论基础

#### 6.2.1 正常Modbus通信特征

基于 Wool (2013) 的研究，正常 HMI-PLC 通信具有以下特征：

> "HMI-PLC communications are **extremely regimented** with few human-initiated actions. A key assumption is that the communications are **highly periodic**: the HMI repeatedly polls every PLC at a **fixed frequency** and issues a **repeating sequence of commands**."

| 特征 | 正常通信 | 异常/攻击 |
|------|---------|----------|
| 规律性 | Extremely regimented（极其规范） | 打破规律性 |
| 周期性 | Highly periodic（高度周期性） | 非周期性突发 |
| 命令序列 | Repeating sequence（重复序列） | 异常序列 |
| 人工干预 | Few human-initiated actions | 大量异常操作 |

#### 6.2.2 数据集攻击特征

根据 CIC Modbus 2023 数据集论文 (Boakye-Boateng et al., 2023)：

- 攻击类型包括：reconnaissance, query flooding, **brute force write**, baseline replay 等
- 正常操作逻辑：
  - IED: "periodically change voltage values randomly or when request received"
  - SCADA: "tap-change based on values received from IED"

**推论**：正常写操作是"响应式"的（读取→分析→写入→验证），而非"连续式"的。

### 6.3 架构设计

```
┌─────────────────────────────────────────────────────────────────────┐
│  两层分层规则架构                                                    │
│                                                                     │
│  设计依据: Wool (2013) - "Communications are highly periodic with    │
│           repeating sequence of commands"                           │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 1: 高置信规则（直接报警）                                      │
│  ├── R1: external_ip_present == 1                                   │
│  │       依据: 工控网络隔离原则，外部IP是入侵指标                     │
│  │                                                                  │
│  └── R3: write_without_read_ratio == 1                              │
│          依据: 违反"highly periodic polling"模式                     │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 2: 中置信规则                                                 │
│  └── R2: consecutive_write_max > 3                                  │
│          依据: 违反"repeating sequence"模式，Brute Force Write特征   │
│          敏感性分析: [1, 3, 5, 10, 15]                               │
├─────────────────────────────────────────────────────────────────────┤
│  组合逻辑:                                                          │
│  rule_attack = Layer1 OR Layer2 = (R1) OR (R3) OR (R2)              │
│  final_prediction = rule_attack OR ml_attack                        │
└─────────────────────────────────────────────────────────────────────┘
```

**设计原理**：
- 规则层捕获高置信度攻击特征，提升 Recall
- ML层处理复杂情况，保持整体性能
- 两者取并集（OR），最大化攻击检测覆盖

### 6.4 规则定义与依据

#### 规则汇总表

| 规则ID | 规则条件 | 置信层 | 理论依据 | 来源 |
|--------|---------|--------|---------|------|
| **R1** | `external_ip_present == 1` | 高 | 工控网络隔离原则，外部IP是入侵指标 | Modbus规范 + 数据集设计 |
| **R2** | `consecutive_write_max > 3` | 中 | 违反"repeating sequence"模式 | Wool (2013) |
| **R3** | `write_without_read_ratio == 1` | 高 | 违反"highly periodic polling"模式 | Wool (2013) |

#### R1: 外部IP检测

```
条件: external_ip_present == 1
置信度: 高
依据:
├── 工控网络应与外部网络隔离
├── 数据集中 185.175.0.7 明确标记为 "External Attacker"
└── Modbus规范中合法设备地址为 1-247

注意: 
├── 在 Compromised-SCADA/IED 场景中可能不触发（模拟内部攻击）
└── 保留此规则是为了方法完整性和实际部署价值
```

#### R2: 连续写入检测

```
条件: consecutive_write_max > 3
置信度: 中
依据:
├── Wool (2013): 正常通信是 "repeating sequence of commands"
├── 正常DCS操作遵循 "读取→分析→写入→验证" 循环
├── 连续多次写入打破了周期性规律
├── Brute Force Write 攻击的核心特征是连续大量写入
└── 阈值3与项目内部 write_burst_count 定义一致

敏感性分析范围: [1, 3, 5, 10, 15]
```

#### R3: 纯写操作检测

```
条件: write_without_read_ratio == 1
置信度: 高
依据:
├── Wool (2013): "HMI repeatedly polls every PLC"
├── 正常通信包含大量周期性读取（轮询）
├── 时间窗口内完全没有读操作是极度异常
└── 与 "highly periodic polling" 特征完全矛盾

特点: 二值条件，无阈值选择问题
```

#### 未采用的规则

| 规则 | 条件 | 未采用原因 |
|------|------|-----------|
| R4 | `fc_write_ratio > 阈值` | 与R3功能重叠，R3更严格且无阈值问题 |

### 6.5 实验步骤

#### 步骤1: FN样本特征分析

从 Phase 2 测试集提取 132 个被 RF 漏检的 Attack 样本，分析特征分布：

| 待分析特征 | 分析目的 |
|-----------|---------|
| `external_ip_present` | 是否存在外部IP攻击 |
| `consecutive_write_max` | 连续写入次数分布 |
| `write_without_read_ratio` | 纯写操作窗口比例 |
| `fc_write_ratio` | 写操作比例分布 |
| `write_burst_count` | 写入突发次数 |
| `operation_sequence_entropy` | 操作序列复杂度 |

**目标**：验证规则设计能否捕获被ML漏检的攻击模式。

#### 步骤2: 规则层实现

```python
def apply_rules(df):
    """应用规则层检测"""
    # Layer 1: 高置信规则
    r1 = df['external_ip_present'] == 1
    r3 = df['write_without_read_ratio'] == 1
    
    # Layer 2: 中置信规则
    r2 = df['consecutive_write_max'] > 3  # 默认阈值，敏感性分析会测试其他值
    
    # 组合
    rule_attack = r1 | r2 | r3
    return rule_attack, {'R1': r1, 'R2': r2, 'R3': r3}
```

#### 步骤3: 规则层单独评估

在测试集上评估规则层性能：

| 指标 | 说明 |
|------|------|
| Rule Recall | 规则在Attack样本上的命中率 |
| Rule FPR | 规则在Normal样本上的误报率 |
| 每条规则触发统计 | R1/R2/R3各自的触发次数和分布 |
| FN覆盖率 | 规则能捕获多少Phase 2的FN样本 |

#### 步骤4: 混合架构评估

组合规则层和ML层的决策：

```python
# 组合决策逻辑
final_prediction = rule_attack | ml_attack  # OR组合
```

计算组合后的指标，与纯RF对比：

| 对比项 | 纯RF (Phase 2) | 混合架构 | 变化 |
|--------|---------------|---------|------|
| Recall | 88.13% | ? | ? |
| Precision | 94.96% | ? | ? |
| F1-Score | 91.42% | ? | ? |

#### 步骤5: 阈值敏感性分析

对 R2 的阈值进行敏感性分析：

| 阈值 | 分析范围 | 目的 |
|------|---------|------|
| `consecutive_write_max` | **[1, 3, 5, 10, 15]** | 展示阈值对Recall-Precision的影响 |

**分析内容**：
- 不同阈值下的 Recall、Precision、F1 变化
- 绘制 Recall-Precision trade-off 曲线
- 确定最佳工程部署阈值建议

### 6.6 预期结果

| 场景 | 预期结果 | 论文结论 |
|------|---------|---------|
| **规则有效** | Recall: 88%→92%+，Precision略降 | 两层架构实现更优trade-off |
| **规则部分有效** | Recall小幅提升(1-3%) | 规则层作为ML的补充手段 |
| **规则无显著效果** | 被漏检攻击无明显规则特征 | 说明ML在复杂攻击检测中不可替代 |

> ⚠️ **无论哪种结果都有论文价值，关键是诚实报告和深入分析**

### 6.7 论文表述模板

```markdown
## Rule-based Detection Layer

### Theoretical Foundation

The rule-based layer is designed based on the well-established characteristics 
of normal Modbus communications. Wool (2013) demonstrated that HMI-PLC 
communications in SCADA systems are "extremely regimented" and "highly 
periodic," with the HMI issuing "repeating sequence of commands" at fixed 
frequencies. This periodicity provides a foundation for detecting anomalies 
that deviate from normal operational patterns.

### Rule Definitions

| Rule | Condition | Rationale |
|------|-----------|-----------|
| R1 | external_ip_present = 1 | External IP access violates ICS network isolation |
| R2 | consecutive_write_max > 3 | Consecutive writes violate "repeating sequence" pattern |
| R3 | write_without_read_ratio = 1 | Pure write windows violate "periodic polling" pattern |

### Threshold Selection

The threshold for R2 is determined based on:
1. The "read-analyze-write-verify" operational cycle in DCS systems
2. Alignment with the internal definition of write_burst_count (>3)
3. Sensitivity analysis across threshold values [1, 3, 5, 10, 15]
```

### 6.8 输出文件

| 文件 | 说明 |
|------|------|
| `results/tables/phase3_fn_analysis.csv` | FN样本特征分析 |
| `results/tables/phase3_rule_evaluation.csv` | 规则层评估结果（含各规则触发统计） |
| `results/tables/phase3_two_layer_results.csv` | 两层架构对比结果 |
| `results/tables/phase3_threshold_sensitivity.csv` | R2阈值敏感性分析 |
| `results/figures/fn_feature_distribution.png` | FN样本特征分布图 |
| `results/figures/two_layer_comparison.png` | 架构性能对比图 |
| `results/figures/threshold_sensitivity.png` | 阈值敏感性分析图 |

---

## 七、E3: 算法对比实验

### 7.1 实验目的

对比 RF 与其他 ML 算法的性能和效率，证明轻量级模型的适用性。

### 7.2 对比算法

| 算法 | 类型 | 用途 |
|------|------|------|
| **Random Forest** | 集成学习 | **主模型** (Phase 2基线) |
| Decision Tree | 单棵树 | 最轻量Baseline |
| Logistic Regression | 线性模型 | 线性Baseline |
| XGBoost | 集成学习 | 高性能集成方法对比 |
| MLP (2-3层) | 深度学习 | 证明轻量级优势 |

### 7.3 模型配置

```python
# Random Forest (Phase 2 配置)
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1
}

# Decision Tree
DT_PARAMS = {
    'max_depth': None,
    'class_weight': 'balanced',
    'random_state': 42
}

# Logistic Regression
LR_PARAMS = {
    'class_weight': 'balanced',
    'max_iter': 1000,
    'random_state': 42
}

# XGBoost
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'scale_pos_weight': 21.7,  # 处理类别不平衡
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

# MLP
MLP_PARAMS = {
    'hidden_layer_sizes': (64, 32),
    'activation': 'relu',
    'max_iter': 200,
    'random_state': 42
}
```

### 7.4 评估指标与方式

#### 评估方式
- **训练集 5-fold CV**：报告均值±标准差（评估模型稳定性）
- **测试集最终评估**：使用全部训练集训练，在测试集上报告最终性能

#### 性能指标
- Accuracy, Precision, Recall, F1-Score, AUC-ROC

#### 效率指标
- 训练时间
- 推理时间 (ms/样本)
- 模型大小 (MB)

### 7.5 预期结论

- RF 与 XGBoost 性能相当
- RF 推理速度比 MLP 快 10x+
- 证明轻量级模型在工控场景的适用性

### 7.6 输出文件

| 文件 | 说明 |
|------|------|
| `results/tables/phase3_algorithm_comparison.csv` | 算法对比结果 |
| `results/figures/algorithm_performance.png` | 性能对比图 |
| `results/figures/inference_time_comparison.png` | 推理时间对比图 |

---

## 八、统一配置参数

### 8.1 随机种子与交叉验证

```python
RANDOM_SEED = 42
N_FOLDS = 5  # 5-fold Stratified Cross-Validation
```

### 8.2 模型评估

```python
# 正类标签
POS_LABEL = 'Attack'

# 评估指标
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auc_roc']
```

### 8.3 图表样式

```python
# 沿用 Phase 2 样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 保存参数
SAVE_PARAMS = {'dpi': 150, 'bbox_inches': 'tight'}
```

---

## 九、输出文件汇总

### 9.1 Notebook

| 文件 | 说明 |
|------|------|
| `notebooks/09_Phase3_核心验证.ipynb` | Phase 3 完整实验代码 |

### 9.2 结果表格 (`results/tables/`)

| 文件 | 实验 |
|------|------|
| `phase3_ablation_results.csv` | E6 消融实验 |
| `phase3_cross_scenario_results.csv` | E7 跨场景泛化 |
| `phase3_fn_analysis.csv` | E4 FN分析 |
| `phase3_rule_evaluation.csv` | E4 规则评估 |
| `phase3_two_layer_results.csv` | E4 两层架构 |
| `phase3_threshold_sensitivity.csv` | E4 阈值敏感性 |
| `phase3_algorithm_comparison.csv` | E3 算法对比 |

### 9.3 结果图表 (`results/figures/`)

| 文件 | 实验 |
|------|------|
| `ablation_comparison.png` | E6 性能对比 |
| `dcs_feature_contribution.png` | E6 DCS特征贡献 |
| `cross_scenario_comparison.png` | E7 泛化对比 |
| `fn_feature_distribution.png` | E4 FN特征分布 |
| `two_layer_comparison.png` | E4 架构对比 |
| `threshold_sensitivity.png` | E4 阈值敏感性 |
| `algorithm_performance.png` | E3 性能对比 |
| `inference_time_comparison.png` | E3 推理时间 |

---

## 十、规则设计理论依据

### 10.1 文献调研结果

#### 10.1.1 核心参考文献

| 来源 | 标题/内容 | 关键贡献 |
|------|----------|---------|
| **Wool (2013)** | "Accurate modeling of Modbus/TCP for intrusion detection in SCADA systems" | 正常通信特征："highly periodic", "repeating sequence" |
| **CIC Dataset (2023)** | Boakye-Boateng et al., PST 2023 | 攻击类型定义，包括 Brute Force Write |
| **DPSTele** | Modbus Poll Basics | 轮询频率："poll takes place many times per second" |
| **Modbus.org** | Modbus Application Protocol Specification | 协议规范，FC定义，PDU限制 |

#### 10.1.2 正常Modbus通信特征（Wool 2013）

```
关键引用:
"HMI-PLC communications are extremely regimented with few human-initiated 
actions. A key assumption is that the communications are highly periodic: 
the HMI repeatedly polls every PLC at a fixed frequency and issues a 
repeating sequence of commands."

实验验证:
"Perfect matches of the model to the traffic were observed for five of 
the seven PLCs tested without a single false alarm over 111 h of operation."
```

**推论**：
1. 正常通信高度周期性 → 非周期性操作是异常
2. 命令序列重复 → 打破序列规律是异常
3. 人工操作很少 → 大量操作是异常

#### 10.1.3 数据集攻击描述（CIC 2023）

```
攻击类型:
"The attacks are reconnaissance, query flooding, loading payloads, 
delay response, modify length parameters, false data injection, 
stacking Modbus frames, brute force write and baseline replay."

正常操作逻辑:
- IED: "periodically change voltage values randomly or when request received"
- SCADA: "tap-change based on values received from IED"
```

**推论**：
- 正常写操作是响应式的（收到请求后执行）
- Brute Force Write 攻击是主动式连续写入

### 10.2 规则阈值依据

#### R1: `external_ip_present == 1`

| 依据类型 | 内容 |
|---------|------|
| 协议规范 | Modbus地址1-247为合法设备 |
| 数据集设计 | 185.175.0.7 明确标记为 External Attacker |
| 工控安全原则 | DCS网络应隔离，外部IP访问是入侵信号 |

#### R2: `consecutive_write_max > 3`

| 依据类型 | 内容 |
|---------|------|
| 理论依据 | Wool (2013): 正常通信是 "repeating sequence"，包含读写交替 |
| 业务逻辑 | DCS操作遵循"读取→分析→写入→验证"循环 |
| 项目一致性 | 与 `write_burst_count` 定义一致（连续写入>3次） |
| 攻击特征 | Brute Force Write 的核心是连续大量写入 |

**阈值选择**：
- 阈值 > 3 表示连续4次以上写入
- 敏感性分析范围：[1, 3, 5, 10, 15]

#### R3: `write_without_read_ratio == 1`

| 依据类型 | 内容 |
|---------|------|
| 理论依据 | Wool (2013): "HMI repeatedly polls every PLC" |
| 业务逻辑 | 正常通信必然包含周期性轮询（读操作） |
| 特征明确 | 窗口内只有写无读，与周期性轮询完全矛盾 |

**阈值选择**：
- 二值条件（== 1），无阈值问题
- 比 `fc_write_ratio > 阈值` 更严格且更明确

### 10.3 已确定的规则配置

```python
# E4 两层架构规则配置
RULE_CONFIG = {
    # Layer 1: 高置信规则
    'R1': {
        'feature': 'external_ip_present',
        'condition': '== 1',
        'confidence': 'high',
        'rationale': 'External IP violates ICS network isolation'
    },
    'R3': {
        'feature': 'write_without_read_ratio',
        'condition': '== 1',
        'confidence': 'high',
        'rationale': 'Violates highly periodic polling pattern (Wool 2013)'
    },
    
    # Layer 2: 中置信规则
    'R2': {
        'feature': 'consecutive_write_max',
        'condition': '> 3',  # 默认阈值
        'confidence': 'medium',
        'rationale': 'Violates repeating sequence pattern (Wool 2013)',
        'sensitivity_range': [1, 3, 5, 10, 15]
    }
}

# 组合逻辑
RULE_COMBINATION = 'R1 OR R2 OR R3'
```

### 10.4 阈值设计原则

```
1. 业务依据优先：每条规则基于工控业务逻辑和学术文献
2. 保守原则：宁可漏报（交给ML处理），不能大量误报
3. 敏感性分析：展示阈值选择对结果的影响
4. 透明报告：论文中说明阈值来源和合理性
5. 可解释性：规则具有明确的业务含义，便于运维人员理解
```

---

## 十一、时间规划

| 阶段 | 实验 | 预计耗时 |
|------|------|---------|
| Day 1 | E6 DCS特征消融 | 2-3小时 |
| Day 2 | E7 跨场景泛化（含反向补充） | 2-3小时 |
| Day 3 | E4 两层架构验证 | 3-4小时 |
| Day 4 | E3 算法对比 | 2-3小时 |
| Day 5 | 结果整理、文档更新 | 2小时 |

---

## 十二、参考文档

### 12.1 项目内部文档

- `docs/PROJECT_DESIGN.md` - 项目总设计文档
- `docs/PHASE1_SUMMARY.md` - Phase 1 执行总结
- `docs/PHASE2_SUMMARY.md` - Phase 2 执行总结
- `src/feature_extractor.py` - 特征分组定义

### 12.2 关键参考文献

#### 数据集来源

```
Boakye-Boateng, K., Ghorbani, A. A., & Lashkari, A. H. (2023). 
"Securing Substations with Trust, Risk Posture and Multi-Agent Systems: 
A Comprehensive Approach"
20th International Conference on Privacy, Security and Trust (PST), 
Copenhagen, Denmark.
数据集: https://www.unb.ca/cic/datasets/modbus-2023.html
```

#### 规则设计理论依据

```
Wool, A. (2013). 
"Accurate modeling of Modbus/TCP for intrusion detection in SCADA systems"
International Journal of Critical Infrastructure Protection, Vol. 6.
关键贡献: 正常Modbus通信特征 - "highly periodic", "repeating sequence"
```

#### 协议规范

```
Modbus Organization. 
"Modbus Application Protocol Specification V1.1b3"
https://www.modbus.org/
```

#### 工控安全参考

```
NIST SP 800-82. 
"Guide to Industrial Control Systems (ICS) Security"

IEC 62443. 
"Industrial communication networks - Network and system security"
```

---

*文档创建时间: 2026-01-14*  
*最后更新: 2026-01-14*
