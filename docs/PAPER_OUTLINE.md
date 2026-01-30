# 论文大纲

> **论文标题**: Feature Analysis and Lightweight Detection for Modbus-based IIoT Systems: A Study on CIC Modbus Dataset 2023
> 
> **目标会议**: EI检索会议
> 
> **截稿日期**: 2026年2月15日

---

## Abstract (约200词)

**背景**: IIoT系统中Modbus协议的安全威胁

**问题**: 
- CIC Modbus 2023是新发布的公开数据集，缺乏系统性分析
- 现有检测方法复杂度高，不适合边缘部署

**方法**: 
- 系统分析CIC Modbus 2023数据集特征
- 设计44维特征工程（协议层+时序层+操作模式）
- 对比5种ML算法和简单规则基线

**结果**:
- 发现`consecutive_write_max`是关键检测特征
- XGBoost达到F1=96.17%，推理时间仅0.001ms
- 简单规则(cwm>0)在特定场景达到F1=1.0，但误报率较高
- ML方法通过减少65-80%误报显著提升实用性

**结论**: 
- 揭示数据集攻击模式特性
- 为不同应用场景提供检测方案选择指南

**Keywords**: Modbus protocol, IIoT security, anomaly detection, feature analysis, lightweight detection

---

## 1. Introduction (约1页)

### 1.1 Background and Motivation

**IIoT安全背景**:
- Industry 4.0中IIoT系统的广泛应用
- SCADA/DCS系统面临的网络安全威胁
- 近年来工控系统攻击事件（可引用Stuxnet, BlackEnergy等）

**Modbus协议安全问题**:
- Modbus协议设计于1979年，缺乏安全机制
- 无认证、无加密、无完整性校验
- 攻击者可轻易进行重放、伪造、暴力写入等攻击

**研究动机**:
- CIC Modbus 2023是加拿大网络安全研究所发布的新数据集
- 该数据集缺乏系统性的特征分析和检测方法研究
- 现有深度学习方法计算复杂度高，不适合IIoT边缘部署

### 1.2 Research Questions

1. CIC Modbus 2023数据集的攻击模式和特征分布是什么？
2. 哪些特征对Modbus异常检测最有效？
3. 简单规则与ML方法的性能差异和适用场景是什么？
4. 如何设计适合IIoT边缘部署的轻量级检测方案？

### 1.3 Contributions

本文的主要贡献：

1. **数据集系统分析**: 首次对CIC Modbus 2023进行全面的特征分析，揭示其攻击模式特性
2. **关键特征识别**: 发现`consecutive_write_max`是区分攻击与正常流量的关键特征
3. **方法对比研究**: 系统对比简单规则、传统ML和神经网络在该数据集上的表现
4. **轻量级方案**: 提出适合IIoT边缘部署的检测方案（推理时间<0.01ms）
5. **实践指南**: 为不同应用场景提供检测方法选择建议

### 1.4 Paper Organization

Section 2介绍相关工作；Section 3描述数据集和特征工程；Section 4介绍检测方法；Section 5展示实验结果；Section 6进行讨论；Section 7总结全文。

---

## 2. Related Work (约0.8页)

### 2.1 IIoT/ICS Security Threats

- 工控系统安全威胁综述
- SCADA系统攻击案例
- IIoT安全挑战

### 2.2 Modbus Protocol Security

- Modbus协议漏洞分析
- Modbus攻击类型分类
  - 侦察攻击 (Reconnaissance)
  - 拒绝服务攻击 (DoS)
  - 写操作攻击 (Write attacks)
  - 重放攻击 (Replay attacks)
- 现有Modbus安全研究

### 2.3 ML-based Anomaly Detection for ICS

- 传统机器学习方法 (RF, SVM, etc.)
- 深度学习方法 (LSTM, CNN, Autoencoder)
- 公开数据集使用情况
- **研究空白**: CIC Modbus 2023数据集的系统性分析

---

## 3. Dataset and Feature Engineering (约1.5页)

### 3.1 CIC Modbus Dataset 2023

#### 3.1.1 Dataset Overview

- 数据来源：Canadian Institute for Cybersecurity
- 实验环境：Docker模拟的SCADA系统
- 数据格式：PCAP网络流量

**表格: 数据集统计**

| Scenario | Files | Windows | Normal | Attack |
|----------|-------|---------|--------|--------|
| Benign | 19 | 44,821 | 44,821 | 0 |
| Compromised-SCADA | 17 | 38,799 | 33,681 | 5,118 |
| Compromised-IED | 6 | 34,582 | 34,483 | 99 |
| External | 1 | 7 | 6 | 1 |
| **Total** | **43** | **118,209** | **112,991** | **5,218** |

#### 3.1.2 Attack Types

- Brute Force Write (>99%)
- Replay attacks
- Reconnaissance

#### 3.1.3 Labeling Strategy

- External场景：基于攻击者IP (185.175.0.7)
- Compromised场景：基于写操作 (Function Code 5/6/15/16)

### 3.2 Feature Engineering

#### 3.2.1 Time Window Design

- 15秒时间窗口
- ≥30包最小阈值过滤
- 稀疏工控流量处理

#### 3.2.2 Feature Categories

**表格: 44维特征分类**

| Category | Count | Examples |
|----------|-------|----------|
| Protocol | 13 | fc_write_ratio, fc_read_ratio, txid_std |
| Temporal | 9 | packet_rate, interval_mean, burst_count |
| Operation Pattern | 22 | consecutive_write_max, write_without_read_ratio |

#### 3.2.3 Key Feature: consecutive_write_max

- 定义：时间窗口内最大连续写操作次数
- 计算方法
- DCS业务含义

### 3.3 Dataset Characteristics Analysis

**⚠️ 关键发现（诚实报告）**:

```
SCADA场景: Attack cwm>0 = 100%, Normal cwm>0 = 0%
IED场景:   Attack cwm>0 = 100%, Normal cwm>0 = 0%
```

- 简单特征`consecutive_write_max > 0`可完美区分SCADA/IED场景中的Attack和Normal
- 该特性揭示了数据集攻击模式的相对简单性
- 对后续研究具有重要参考价值

---

## 4. Methodology (约0.8页)

### 4.1 Problem Formulation

- 二分类问题：Normal vs Attack
- 输入：44维特征向量
- 输出：类别标签和置信度

### 4.2 Rule-based Baseline

**简单规则**:
```
if consecutive_write_max > 0:
    return Attack
else:
    return Normal
```

- 零漏检（Recall=100%）
- 计算开销极低

### 4.3 Machine Learning Methods

| Algorithm | Key Parameters |
|-----------|---------------|
| Random Forest | n_estimators=100, class_weight='balanced' |
| Decision Tree | max_depth=None, class_weight='balanced' |
| Logistic Regression | max_iter=1000, class_weight='balanced' |
| XGBoost | n_estimators=100, scale_pos_weight=21.7 |
| MLP | hidden_layers=(100,50), max_iter=500 |

### 4.4 Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- 5-Fold Cross-Validation
- 推理时间和模型大小

---

## 5. Experiments and Results (约2页)

### 5.1 Experimental Setup

- 数据划分：70% train / 15% val / 15% test（按PCAP文件划分）
- 类别不平衡处理：class_weight='balanced'
- 硬件环境

### 5.2 Main Detection Results

**表格: 算法性能对比**

| Method | Accuracy | Precision | Recall | F1 | Inference(ms) |
|--------|----------|-----------|--------|-----|---------------|
| Simple Rule (cwm>0) | 97.31% | 67.64% | **100%** | 80.70% | 0.0001 |
| Random Forest | 99.21% | 94.84% | 90.83% | 92.79% | 0.0065 |
| XGBoost | **99.58%** | **98.86%** | 93.61% | **96.17%** | 0.0011 |
| Decision Tree | 99.07% | 93.36% | 91.10% | 93.36% | 0.0002 |
| Logistic Regression | 98.07% | 81.45% | 98.92% | 81.45% | 0.0001 |
| MLP | 97.15% | 81.01% | 69.06% | 81.01% | 0.0150 |

**图: 性能对比雷达图**

### 5.3 ML vs Simple Rule Analysis

**ML相对于简单规则的增益**:

| Model | F1 Gain | Relative Improvement |
|-------|---------|---------------------|
| Random Forest | +12.09% | +14.98% |
| XGBoost | +15.47% | **+19.17%** |

**ML的核心价值**: 通过牺牲少量Recall换取大幅Precision提升

### 5.4 Feature Importance Analysis

**表格: Top-10重要特征**

| Rank | Feature | Type | Importance |
|------|---------|------|------------|
| 1 | write_without_read_ratio | Operation | 0.15 |
| 2 | fc_write_ratio | Protocol | 0.12 |
| 3 | consecutive_write_max | Operation | 0.10 |
| ... | ... | ... | ... |

**发现**: Top-10中6个与写操作相关的特征

### 5.5 Feature Ablation Study

**表格: 特征消融实验**

| Feature Set | Features | CV F1 | Test F1 |
|-------------|----------|-------|---------|
| Protocol only | 13 | 95.34% | 92.21% |
| Temporal only | 9 | 80.94% | 85.37% |
| Operation only | 22 | 70.62% | 80.56% |
| Protocol + Temporal | 22 | 95.14% | 93.35% |
| All features | 44 | 94.91% | 92.79% |

**发现**: 协议层特征已能达到较高性能

### 5.6 Error Analysis

**表格: 误判数量对比**

| Method | FN | FP | Total Errors |
|--------|----|----|--------------|
| Simple Rule | 0 | 532 | 532 |
| Random Forest | 21 | 185 | 206 |
| XGBoost | 5 | 108 | 113 |

**FP来源分析**:
- 全部532个规则FP来自benign场景
- 原因：正常运维中的少量连续写操作

**ML纠正能力**:
- RF纠正65.2%的规则FP
- XGBoost纠正79.7%的规则FP

**图: 误判分析对比图**

### 5.7 Scenario-wise Performance

**表格: 分场景性能（简单规则）**

| Scenario | F1 |
|----------|-----|
| SCADA | **1.0000** |
| IED | **1.0000** |
| Benign | - (FP rate: 8.96%) |

**发现**: 简单规则在攻击场景(SCADA/IED)达到完美检测，但对正常场景(benign)有较高误报

### 5.8 Lightweight Validation

**表格: 轻量级指标**

| Model | Inference Time (ms) | Model Size (MB) | Throughput |
|-------|---------------------|-----------------|------------|
| Simple Rule | 0.0001 | 0.00 | >10M/s |
| XGBoost | 0.0011 | 0.24 | >900K/s |
| Decision Tree | 0.0002 | 0.08 | >5M/s |
| Random Forest | 0.0065 | 9.99 | >150K/s |

**结论**: 所有方法均满足IIoT边缘部署需求

---

## 6. Discussion (约0.8页)

### 6.1 Key Findings

1. **数据集特性**: CIC Modbus 2023的攻击模式相对简单，`cwm>0`可完美区分攻击场景
2. **特征有效性**: 写操作相关特征是最有效的检测指标
3. **ML价值**: ML方法的核心价值在于减少误报(65-80%)，而非提升检测率
4. **算法选择**: XGBoost在性能和效率上取得最佳平衡

### 6.2 Practical Recommendations

**表格: 应用场景建议**

| Scenario | Recommended Method | Reason |
|----------|-------------------|--------|
| Critical Infrastructure | Simple Rule | Zero false negatives |
| General Monitoring | XGBoost | Balance detection and FP |
| Resource-constrained | Simple Rule or DT | Minimal overhead |
| High-precision | ML + Manual Review | Reduce alert fatigue |

### 6.3 Limitations

**必须诚实报告**:

1. **数据集局限**: 
   - 攻击类型单一（99%+ Brute Force Write）
   - Attack/Normal在特定特征上完全可分
   
2. **泛化性未验证**: 
   - 结论主要适用于类似攻击模式的场景
   - 复杂攻击场景需进一步验证

3. **标签策略**: 
   - 使用流量特征推断标签
   - 可能存在少量标签噪声

### 6.4 Future Work

- 在更复杂数据集上验证方法泛化性
- 设计针对复杂攻击模式的检测方法
- 研究在线检测和增量学习

---

## 7. Conclusion (约0.3页)

**总结**:
- 首次系统分析CIC Modbus 2023数据集
- 发现`consecutive_write_max`是关键检测特征
- XGBoost达到F1=96.17%，推理时间仅0.001ms
- 提供不同应用场景的检测方案选择指南

**核心贡献**:
- 揭示数据集攻击模式特性
- 量化简单规则与ML方法的性能差异
- 为后续Modbus安全研究提供参考基础

---

## References (约15-20条)

**必引文献**:
1. CIC Modbus 2023数据集原始论文
2. Modbus协议规范
3. 工控系统安全综述
4. 相关ML检测方法

**建议引用方向**:
- ICS/SCADA安全
- Modbus协议安全
- ML-based异常检测
- 工控数据集研究

---

## 图表清单

### 图

| 编号 | 标题 | 内容 |
|------|------|------|
| Fig.1 | System architecture | 检测框架示意图 |
| Fig.2 | Feature distribution | 关键特征分布对比 |
| Fig.3 | Algorithm comparison | 算法性能雷达图 |
| Fig.4 | Error analysis | 误判分析对比图 |
| Fig.5 | Feature importance | Top-10特征重要性 |

### 表

| 编号 | 标题 | 内容 |
|------|------|------|
| Table I | Dataset statistics | 数据集统计 |
| Table II | Feature categories | 特征分类 |
| Table III | Algorithm performance | 算法性能对比 |
| Table IV | ML vs Rule comparison | ML增益分析 |
| Table V | Feature ablation | 消融实验结果 |
| Table VI | Error analysis | 误判统计 |
| Table VII | Practical recommendations | 应用建议 |

---

## 预计篇幅

| Section | Pages |
|---------|-------|
| Abstract | 0.3 |
| Introduction | 1.0 |
| Related Work | 0.8 |
| Dataset and Features | 1.5 |
| Methodology | 0.8 |
| Experiments | 2.0 |
| Discussion | 0.8 |
| Conclusion | 0.3 |
| References | 0.5 |
| **Total** | **~8 pages** |

---

*大纲创建时间: 2026-01-22*
