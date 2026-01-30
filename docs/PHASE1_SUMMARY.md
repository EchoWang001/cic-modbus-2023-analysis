# Phase 1: 数据准备 - 执行总结

> 执行时间：2026年1月9日  
> 状态：✅ 已完成

---

## 一、工作概述

Phase 1 完成了从原始PCAP文件到可用于机器学习的特征数据集的全部数据准备工作。

### 执行步骤

| 步骤 | 名称 | 说明 | 耗时 |
|------|------|------|------|
| Step 1 | PCAP文件解析 | 使用Scapy解析43个PCAP文件，提取Modbus包 | ~3小时 |
| Step 2 | 应用标签策略 | 根据IP和写操作标记Attack/Normal | ~5分钟 |
| Step 3 | 15秒窗口特征提取 | 计算44个特征 | ~30分钟 |
| Step 3.5 | 最小包数过滤 | 过滤packet_count < 30的窗口 | <1分钟 |
| Step 4 | 数据集划分 | 按PCAP文件划分训练/验证/测试集 | <1分钟 |

---

## 二、数据文件位置

### 2.1 处理后数据 (`data/processed/`)

| 文件名 | 大小 | 说明 |
|--------|------|------|
| `packets_raw.csv` | ~500 MB | 原始解析的Modbus包数据（CSV格式备份） |
| `packets_labeled.parquet` | 103.38 MB | 标记后的包级数据 |
| `features_15s.parquet` | 7.55 MB | 原始窗口特征（未过滤，121,934个窗口） |
| `features_15s_filtered.parquet` | 7.31 MB | **过滤后窗口特征（主数据，118,209个窗口）** |

### 2.2 数据集划分 (`data/splits/`)

| 文件名 | 大小 | 窗口数 | 说明 |
|--------|------|--------|------|
| `train.parquet` | 5.15 MB | ~83,000 | 训练集 (70%) |
| `val.parquet` | 1.03 MB | ~17,500 | 验证集 (15%) |
| `test.parquet` | 1.26 MB | ~17,700 | 测试集 (15%) |
| `split_info.json` | <1 KB | - | 划分元信息 |

### 2.3 源代码 (`src/`)

| 文件名 | 说明 |
|--------|------|
| `config.py` | 项目配置（路径、参数等） |
| `pcap_parser.py` | PCAP解析模块 |
| `feature_extractor.py` | 特征提取模块（44个特征） |

### 2.4 Notebook (`notebooks/`)

| 文件名 | 说明 |
|--------|------|
| `07_Phase1_数据准备.ipynb` | Phase 1 完整执行记录 |

---

## 三、数据统计

### 3.1 原始数据

| 指标 | 数值 |
|------|------|
| PCAP文件数 | 43 |
| 原始Modbus包数 | 8,478,926 |
| 数据时间跨度 | ~518小时 (约21.5天) |

### 3.2 场景分布

| 场景 | PCAP文件数 | 总时长 | 窗口数（过滤后） |
|------|-----------|--------|----------------|
| Benign | 19 | 187.6h | 44,821 |
| Compromised-SCADA | 17 | 163.1h | 38,799 |
| Compromised-IED | 6 | 157.7h | 34,582 |
| External | 1 | 9.5h | 7 |
| **总计** | **43** | **517.9h** | **118,209** |

### 3.3 过滤效果

| 指标 | 过滤前 | 过滤后 |
|------|--------|--------|
| 窗口数 | 121,934 | 118,209 |
| 保留率 | 100% | 96.9% |
| 最小包数阈值 | - | ≥30 |

### 3.4 标签分布

| 标签 | 窗口数 | 比例 |
|------|--------|------|
| Normal | 112,991 | 95.6% |
| Attack | 5,218 | 4.4% |
| **比例** | **Normal:Attack = 21.7:1** | |

### 3.5 特征统计

| 指标 | 数值 |
|------|------|
| 特征总数 | 44 |
| 协议层特征 | 13 |
| 时序层特征 | 9 |
| DCS业务逻辑特征 | 22 |
| 样本/特征比 | 2686.6:1 |

---

## 四、标签策略

### 4.1 策略说明

| 场景 | 标签规则 |
|------|---------|
| **Benign** | 所有窗口 → Normal |
| **External** | 窗口内有 `src_ip == 185.175.0.7` → Attack，否则 → Normal |
| **Compromised-SCADA/IED** | 窗口内有写操作 (FC 5/6/15/16) → Attack，否则 → Normal |

### 4.2 攻击者IP

```
185.175.0.7 → External Attacker
```

### 4.3 写操作功能码

```
FC 5  - Write Single Coil
FC 6  - Write Single Register
FC 15 - Write Multiple Coils
FC 16 - Write Multiple Registers
```

---

## 五、特征列表

### 5.1 协议层特征 (13个)

| 特征名 | 说明 |
|--------|------|
| fc_distribution_entropy | 功能码分布熵 |
| fc_diversity | 功能码种类数 |
| fc_read_ratio | 读操作比例 |
| fc_write_ratio | 写操作比例 |
| txid_mean | 事务ID均值 |
| txid_std | 事务ID标准差 |
| txid_unique_ratio | 唯一TxID比例 |
| unit_id_diversity | 单元ID多样性 |
| packet_size_mean | 平均包大小 |
| packet_size_std | 包大小标准差 |
| address_range | 访问地址范围 |
| value_mean | 写入值均值 |
| value_std | 写入值标准差 |

### 5.2 时序层特征 (9个)

| 特征名 | 说明 |
|--------|------|
| packet_count | 窗口内包数量 |
| packet_rate | 包速率 (packets/sec) |
| interval_mean | 包间隔均值 |
| interval_std | 包间隔标准差 |
| interval_min | 最小包间隔 |
| interval_max | 最大包间隔 |
| burst_count | 突发次数 |
| burst_intensity | 突发强度 |
| session_count | 会话数量 |

### 5.3 DCS业务逻辑特征 (22个)

#### 设备角色特征 (6个)
- scada_src_ratio, scada_dst_ratio
- ied_src_ratio, ied_dst_ratio
- device_role_entropy, is_typical_scada_to_ied

#### 通信拓扑特征 (5个)
- src_ip_count, dst_ip_count
- comm_pair_count, dominant_pair_ratio, multi_target_ratio

#### 操作模式特征 (6个)
- consecutive_write_max, consecutive_write_mean
- write_burst_count, read_write_alternation
- operation_sequence_entropy, write_without_read_ratio

#### 异常指标特征 (5个)
- external_ip_present, external_ip_packet_ratio
- unknown_ip_count, abnormal_fc_ratio, address_range_exceeded

---

## 六、关键配置参数

```python
# 时间窗口
TIME_WINDOW = 15  # 秒

# 最小包数阈值
MIN_PACKETS_THRESHOLD = 30

# 数据划分比例
SPLIT_RATIOS = {
    'train': 0.70,
    'val': 0.15,
    'test': 0.15
}

# 随机种子
RANDOM_SEED = 42

# 攻击者IP
ATTACKER_IP = '185.175.0.7'

# 写操作功能码
WRITE_FUNCTION_CODES = [5, 6, 15, 16]
```

---

## 七、数据加载示例

### 加载过滤后特征数据

```python
import pandas as pd

# 加载完整特征数据
df_features = pd.read_parquet('data/processed/features_15s_filtered.parquet')
print(f"总窗口数: {len(df_features)}")
```

### 加载训练/验证/测试集

```python
# 加载划分后的数据集
df_train = pd.read_parquet('data/splits/train.parquet')
df_val = pd.read_parquet('data/splits/val.parquet')
df_test = pd.read_parquet('data/splits/test.parquet')

print(f"训练集: {len(df_train)}")
print(f"验证集: {len(df_val)}")
print(f"测试集: {len(df_test)}")
```

### 获取特征名列表

```python
from feature_extractor import get_feature_names

feature_names = get_feature_names()
print(f"特征数: {len(feature_names)}")
```

---

## 八、注意事项

1. **类别不平衡**: Normal:Attack ≈ 21.7:1，训练时需使用 `class_weight="balanced"`
2. **External场景样本少**: 过滤后仅7个窗口，主要依赖Compromised场景进行评估
3. **工控流量稀疏性**: 平均每窗口约70包，已通过≥30包阈值过滤确保特征有效性
4. **数据划分原则**: 按PCAP文件划分，同一文件的窗口不跨集合，防止数据泄漏

---

## 九、下一步工作

**Phase 2: 基础实验**
- 实现Random Forest主模型
- 5-fold交叉验证
- 基础性能评估 (Accuracy, Precision, Recall, F1)

---

*文档生成时间: 2026-01-09*
