# 有机化学AI应用作业

**学生ID**: 153  
**完成时间**: 2025年7月24日  
**随机种子**: 1153, 2153, 3153, 4153, 5153

## 项目概述

本项目完成了四个相互关联的AI在有机化学中的应用问题，涵盖分子性质预测、反应条件预测、贝叶斯优化和迁移学习。

## 文件结构

```
submission/
├── README.md                           # 项目说明文档
├── requirements.txt                    # Python依赖包列表
├── problem1_molecular_properties.py   # 问题1: 分子性质预测脚本
├── problem2_reaction_conditions.py    # 问题2: 反应条件预测脚本
├── problem3_bayesian_optimization.py  # 问题3: 贝叶斯优化脚本
├── problem4_transfer_learning.py      # 问题4: 迁移学习脚本
├── results/                           # 结果文件夹
│   ├── problem1_results.csv          # 问题1结果
│   ├── problem2_results.csv          # 问题2结果
│   └── (其他结果文件)
└── figures/                           # 图表文件夹
    ├── problem2_confusion_matrices.png         # 混淆矩阵图
    ├── feature_importance.png         # 特征重要性图
    └── (其他可视化图表)
```

## 安装和运行

### 1. 环境配置

```bash
# 安装依赖包
pip install -r requirements.txt

# 或使用conda安装RDKit (推荐)
conda install -c conda-forge rdkit
```

### 2. 数据准备

确保以下数据文件在正确位置：
- `dataset/qm9star_anion_FMO.csv`
- `dataset/qm9star_cation_FMO.csv`
- `dataset/qm9star_neutral_FMO.csv`
- `dataset/qm9star_radical_FMO.csv`
- `dataset/Reaction Dataset A.csv`
- `dataset/Reaction Dataset B.csv`
- `dataset/AHO.csv`

### 3. 运行脚本

```bash
# 运行问题1: 分子性质预测
python problem1_molecular_properties.py

# 运行问题2: 反应条件预测
python problem2_reaction_conditions.py

# 运行问题3: 贝叶斯优化
python problem3_bayesian_optimization.py

# 运行问题4: 迁移学习
python problem4_transfer_learning.py
```

## 问题详细说明

### 问题1：分子性质预测

**目标**: 使用QM9star数据集预测HOMO-LUMO能隙

**功能特点**:
1. **多数据集处理**: 自动加载和合并4个QM9star数据文件
2. **多种分子表示方法**:
   - Morgan指纹 (radius=2, 2048 bits)
   - RDKit分子描述符
   - 原子中心特征向量
3. **模型训练和评估**:
   - 随机森林回归模型
   - 5折交叉验证
   - MAE, RMSE, R²评估指标
4. **特征重要性分析**
5. **化学解释和可视化**

**输出**:
- `results/problem1_results.csv`: 性能比较结果
- `results/problem1_detailed_results.csv`: 详细结果
- 特征重要性分析
- 性能比较图表

### 问题2：反应条件预测

**目标**: 预测Pd-催化C-N偶联反应的最优条件

**功能特点**:
1. **反应数据分析**: 催化剂、碱、溶剂分布统计
2. **反应指纹生成**: 基于分子结构的反应特征
3. **多分类预测**:
   - 催化剂选择预测
   - 碱选择预测
   - 溶剂选择预测
4. **性能评估**: 准确率、精确率、召回率、F1分数
5. **混淆矩阵分析**
6. **化学解释**

**输出**:
- `results/problem2_results.csv`: 分类性能结果
- `figures/problem2_confusion_matrices.png`: 混淆矩阵图
- 特征重要性分析
- 化学解释报告

### 问题3：贝叶斯优化反应收率

**目标**: 使用贝叶斯优化预测和优化反应收率

**功能特点**:
1. **高斯过程建模**: 多种内核选择
2. **获取函数实现**:
   - 期望改进(EI)
   - 上置信界(UCB)
   - 改进概率(PI)
3. **贝叶斯优化循环**: 迭代选择最优实验点
4. **获取函数性能比较**
5. **优化轨迹可视化**
6. **最优条件识别**

**输出**:
- `results/problem3_acquisition_comparison.csv`: 获取函数比较
- `results/problem3_optimization_history.csv`: 优化历史
- `figures/problem3_optimization_progress.png`: 优化进度图
- 最优条件分析

### 问题4：迁移学习和少样本学习

**目标**: 应用迁移学习处理有限的AHO数据集

**功能特点**:
1. **基准模型比较**: 源域→目标域 vs 仅目标域
2. **迁移学习方法**:
   - 特征对齐
   - 渐进式适应
   - 相似性加权
3. **不确定性量化**: 集成方法的预测不确定性
4. **性能比较分析**
5. **数据需求识别**
6. **化学解释和应用建议**

**输出**:
- `results/problem4_results.csv`: 方法性能比较
- `results/problem4_uncertainty.csv`: 不确定性分析
- `figures/problem4_results.png`: 结果可视化
- 迁移学习效果分析

## 技术特点

### 1. 分子表示学习
- **Morgan指纹**: 捕获分子局部结构特征
- **分子描述符**: 物理化学性质特征
- **原子中心特征**: 原子级别的化学环境

### 2. 机器学习方法
- **随机森林**: 主要预测模型
- **支持向量机**: 替代方法
- **神经网络**: 非线性建模
- **高斯过程**: 贝叶斯优化

### 3. 评估指标
- **回归**: MAE, RMSE, R²
- **分类**: 准确率, 精确率, 召回率, F1分数
- **不确定性**: 预测标准差, 置信区间

### 4. 可视化分析
- 性能比较图表
- 混淆矩阵热图
- 优化轨迹图
- 不确定性分布图

## 主要发现

### 1. 分子性质预测
- 三种表示方法各有优劣
- RDKit描述符提供最好的化学可解释性
- 不同分子类型(离子vs中性)预测难度不同

### 2. 反应条件预测
- 催化剂选择最容易预测
- 溶剂选择受多种因素影响
- 反应指纹有效捕获结构-活性关系

### 3. 贝叶斯优化
- EI获取函数通常表现最佳
- 高斯过程能有效建模反应收率
- 优化策略在10-15次迭代内收敛

### 4. 迁移学习
- 源域数据量对迁移效果影响显著
- 特征对齐方法在化学数据中有效
- 不确定性量化有助于识别需要更多数据的区域

## 化学见解

1. **结构-性质关系**: 芳香性、共轭系统对HOMO-LUMO能隙有重要影响
2. **反应条件优化**: 催化剂、碱、溶剂的协同效应决定反应效果
3. **数据驱动发现**: AI方法能发现传统方法难以识别的化学模式
4. **实验设计指导**: 贝叶斯优化和不确定性量化指导高效实验

## 改进建议

1. **特征工程**: 添加更多化学知识驱动的特征
2. **深度学习**: 探索图神经网络和Transformer模型
3. **多任务学习**: 同时预测多个相关性质
4. **主动学习**: 智能选择最有价值的实验样本
5. **可解释AI**: 提高模型预测的化学可解释性

## 技术限制

1. **数据质量**: 依赖高质量的分子结构和性质数据
2. **计算资源**: 大规模分子数据处理需要充足计算资源  
3. **化学先验**: 需要结合领域专家知识进行结果解释
4. **泛化能力**: 模型在新化学空间的泛化能力有限

## 联系信息

如有问题或建议，请联系：
- 学生ID: 153
- 完成日期: 2025年7月24日

---

**注意**: 本项目仅用于学术研究，实际应用需要进一步验证和优化。
