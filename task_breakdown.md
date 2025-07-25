# 有机化学AI应用作业任务分解

## 总体概述

本作业包含4个相互关联的实际问题，旨在提供有机化学AI应用的实践经验。每个学生需要使用基于学生ID：153组成的5个特定随机种子：1153, 2153, 3153, 4153, 5153，来确保结果的可重现性。

---

## 问题1：分子性质预测

### 目标
使用QM9star数据集开发和比较多种分子表示方法来预测HOMO-LUMO能隙。

### 主要函数分解

#### 1.1 数据加载和探索模块
```python
# 文件：problem1_molecular_properties.py

def load_individual_qm9star_files(dataset_dir):
    """
    加载QM9star数据集的所有4个文件
    - 读取dataset/qm9star_anion_FMO.csv
    - 读取dataset/qm9star_cation_FMO.csv  
    - 读取dataset/qm9star_neutral_FMO.csv
    - 读取dataset/qm9star_radical_FMO.csv
    - 返回包含4个DataFrame的字典
    """

def merge_qm9star_datasets(datasets_dict):
    """
    合并4个QM9star数据集
    - 添加分子类型标识列（anion, cation, neutral, radical）
    - 合并所有数据到单一DataFrame
    - 检查数据一致性和列名匹配
    - 返回合并后的DataFrame
    """

def analyze_molecular_species_distribution(merged_df):
    """
    分析合并数据集中不同分子类型的分布
    - 统计每种分子类型的数量
    - 分析各类型的HOMO-LUMO能隙分布差异
    - 生成分子类型分布可视化
    - 返回分布分析报告
    """

def explore_molecular_data(df):
    """
    执行探索性数据分析
    - 分析分子结构和能隙分布
    - 生成基本统计信息
    - 按分子类型进行分层分析
    - 返回分析报告字典
    """

def select_representative_subset(df, n_samples=100000):
    """
    为每种分子类型选择代表性子集
    - 参数：合并的分子数据框，每种类型的样本数
    - 按分子类型（anion, cation, neutral, radical）分层采样
    - 确保每种类型都有足够的代表性样本
    - 返回：选定的子集DataFrame，保持类型分布平衡
    """

def balance_molecular_species(df, strategy='equal'):
    """
    平衡不同分子类型的样本数量
    - strategy='equal': 每种类型取相同数量样本
    - strategy='proportional': 按原始比例保持分布
    - strategy='minimum': 以最少类型的数量为准
    - 返回平衡后的DataFrame
    """

def validate_smiles_strings(df, smiles_column):
    """
    识别和处理缺失或无效的SMILES字符串
    - 检查SMILES有效性
    - 处理缺失值
    - 返回清理后的数据
    """
```

#### 1.2 分子表示实现模块
```python
def generate_morgan_fingerprints(smiles_list, radius=2, n_bits=2048):
    """
    实现Morgan指纹
    - 参数：SMILES列表，半径，比特数
    - 返回：指纹矩阵
    """

def calculate_rdkit_descriptors(smiles_list):
    """
    计算RDKit分子描述符
    - 使用RDKit计算标准描述符
    - 返回描述符矩阵
    """

def generate_atom_centered_features(smiles_list):
    """
    生成原子中心特征向量
    - 计算原子级别特征
    - 聚合为分子级别特征
    - 返回特征矩阵
    """

def create_representation_comparison_table(representations_dict):
    """
    创建表示方法比较表
    - 输入：不同表示方法的字典
    - 返回：比较表DataFrame
    """
```

#### 1.3 模型开发和评估模块
```python
def train_random_forest_model(X, y, random_state):
    """
    使用特定表示训练随机森林模型
    - 参数：特征矩阵，目标变量，随机种子
    - 返回：训练好的模型
    """

def perform_cross_validation(X, y, model, cv=5, random_state=None):
    """
    实现交叉验证（5折）
    - 计算MAE, RMSE, R²
    - 返回评估指标字典
    """

def analyze_feature_importance(model, feature_names):
    """
    分析最佳模型的特征重要性
    - 提取特征重要性
    - 生成可视化
    - 返回重要性分析
    """

def compare_model_performance(results_dict):
    """
    比较不同表示方法的模型性能
    - 生成性能比较表
    - 分析哪种分子表示最有效
    - 按分子类型进行分层性能分析
    - 返回比较分析
    """

def analyze_species_specific_performance(model, X_test, y_test, species_labels):
    """
    分析不同分子类型的模型性能差异
    - 按分子类型（anion, cation, neutral, radical）分别评估
    - 识别模型对哪种分子类型预测效果最好/最差
    - 分析不同类型分子的预测难度差异
    - 返回分子类型特异性性能报告
    """
```

---

## 问题2：反应条件预测

### 目标
使用反应数据集A构建多分类模型来预测最优反应条件。

### 主要函数分解

#### 2.1 反应数据处理模块
```python
# 文件：problem2_reaction_conditions.py

def load_reaction_dataset_a(file_path):
    """
    加载反应数据集A
    - 读取datasets/Reaction_Dataset_A.csv
    - 返回反应数据DataFrame
    """

def analyze_reaction_conditions_distribution(df):
    """
    分析催化剂、碱和溶剂的分布
    - 统计各类别的频率
    - 生成分布图表
    - 返回分布分析
    """

def encode_categorical_variables(df, categorical_columns):
    """
    处理分类变量并创建适当的编码
    - 实现标签编码或独热编码
    - 返回编码后的数据
    """
```

#### 2.2 反应指纹开发模块
```python
def implement_reaction_fingerprints(reactant_smiles):
    """
    使用反应物分子指纹实现反应指纹
    - 计算反应物的分子指纹
    - 组合成反应指纹
    - 返回反应指纹矩阵
    """

def create_condition_feature_vectors(catalyst_list, base_list, solvent_list):
    """
    创建条件特征向量（催化剂、碱、溶剂组合）
    - 编码反应条件
    - 返回条件特征矩阵
    """

def combine_molecular_condition_features(mol_features, condition_features):
    """
    适当地组合分子和条件特征
    - 特征连接或其他组合方法
    - 返回组合特征矩阵
    """
```

#### 2.3 分类模型实现模块
```python
def train_catalyst_prediction_model(X, y_catalyst, random_state):
    """
    训练催化剂选择预测模型
    - 多分类模型训练
    - 返回训练好的模型
    """

def train_base_prediction_model(X, y_base, random_state):
    """
    训练最佳碱选择模型
    - 多分类模型训练
    - 返回训练好的模型
    """

def train_solvent_prediction_model(X, y_solvent, random_state):
    """
    训练合适溶剂预测模型
    - 多分类模型训练
    - 返回训练好的模型
    """

def evaluate_classification_models(models_dict, X_test, y_test_dict):
    """
    使用适当的评估指标评估模型
    - 计算accuracy, precision, recall, F1-score
    - 生成混淆矩阵
    - 返回评估结果
    """

def interpret_model_predictions(models_dict, feature_names):
    """
    对模型预测进行化学解释
    - 分析重要特征
    - 提供化学见解
    - 返回解释报告
    """
```

---

## 问题3：反应收率的贝叶斯优化

### 目标
使用反应数据集B实现贝叶斯优化来预测和优化反应收率。

### 主要函数分解

#### 3.1 高斯过程模型设置模块
```python
# 文件：problem3_bayesian_optimization.py

def load_doyle_dataset(file_path):
    """
    从datasets/Reaction_Dataset_B.csv加载Doyle数据集
    - 读取数据
    - 返回DataFrame
    """

def implement_gaussian_process_regressor(kernel_type='rbf'):
    """
    使用适当核函数实现高斯过程回归器
    - 选择和配置核函数
    - 返回GPR模型
    """

def preprocess_features(X):
    """
    处理特征缩放和预处理
    - 标准化或归一化特征
    - 返回预处理后的特征
    """
```

#### 3.2 获取函数实现模块
```python
def implement_expected_improvement(gpr_model, X_observed, y_observed, xi=0.01):
    """
    实现期望改进（EI）获取函数
    - 计算EI值
    - 返回EI函数
    """

def implement_upper_confidence_bound(gpr_model, X_observed, kappa=2.576):
    """
    实现上置信界（UCB）作为替代方案
    - 计算UCB值
    - 返回UCB函数
    """

def compare_acquisition_functions(ei_results, ucb_results):
    """
    比较获取函数性能
    - 分析不同获取函数的效果
    - 返回比较分析
    """
```

#### 3.3 贝叶斯优化循环模块
```python
def initialize_bayesian_optimization(X_space, n_initial=20, random_state=None):
    """
    初始化贝叶斯优化
    - 选择初始20个点
    - 返回初始点和对应的观测值
    """

def bayesian_optimization_loop(gpr_model, acquisition_func, X_space, n_iterations=10):
    """
    实现主动学习循环进行收率优化
    - 迭代选择10个新实验
    - 更新模型
    - 返回优化轨迹
    """

def track_optimization_progress(optimization_history):
    """
    跟踪优化进度和收敛性
    - 记录最佳值变化
    - 生成收敛图
    - 返回进度分析
    """

def analyze_optimal_conditions(optimization_results):
    """
    分析发现的最优反应条件
    - 识别最佳参数组合
    - 提供化学解释
    - 返回最优条件分析
    """
```

---

## 问题4：迁移学习和少样本学习

### 目标
应用迁移学习技术使用来自更大数据集的知识处理有限的AHO数据集。

### 主要函数分解

#### 4.1 基准模型训练模块
```python
# 文件：problem4_transfer_learning.py

def load_aho_datasets(aho_path, oos_path):
    """
    加载AHO数据集
    - 读取datasets/aho.csv
    - 读取datasets/aho_oos.csv
    - 返回两个数据集
    """

def train_baseline_aho_model(X_aho, y_aho, random_state):
    """
    仅使用AHO数据集训练基准模型
    - 训练基础模型
    - 返回训练好的模型
    """

def train_baseline_oos_model(X_oos, y_oos, random_state):
    """
    仅使用样本外（OOS）数据集训练基准模型
    - 训练基础模型
    - 返回训练好的模型
    """

def document_small_dataset_challenges(aho_results, oos_results):
    """
    通过5折交叉验证比较AHO和OOS数据集性能，记录小数据集训练的挑战
    - 执行交叉验证比较
    - 分析性能差异
    - 返回挑战文档
    """
```

#### 4.2 迁移学习实现模块
```python
def train_source_models_on_aho(X_aho, y_aho, model_types=['rf', 'svm', 'nn']):
    """
    在AHO数据集上训练更多模型
    - 训练多种模型类型
    - 返回训练好的模型字典
    """

def fine_tune_on_oos_data(pretrained_models, X_oos, y_oos):
    """
    在有限的OOS数据上调优训练好的模型
    - 实现模型微调
    - 返回微调后的模型
    """

def implement_domain_adaptation(source_data, target_data):
    """
    实现域适应技术
    - 应用域适应方法
    - 返回适应后的模型
    """

def compare_transfer_vs_baseline(baseline_results, transfer_results):
    """
    比较迁移学习与基准性能
    - 性能指标比较
    - 统计显著性测试
    - 返回比较分析
    """
```

#### 4.3 不确定性量化模块
```python
def implement_prediction_uncertainty(models_list, X_test):
    """
    实现预测不确定性估计
    - 计算预测置信区间
    - 返回不确定性估计
    """

def ensemble_methods_uncertainty(models_list, X_test):
    """
    使用集成方法进行不确定性量化
    - 模型集成
    - 方差估计
    - 返回集成不确定性
    """

def bayesian_uncertainty_estimation(model, X_test):
    """
    使用贝叶斯方法估计不确定性
    - 贝叶斯推理
    - 后验分布采样
    - 返回贝叶斯不确定性
    """

def analyze_model_confidence(uncertainty_results, predictions):
    """
    分析模型预测的置信度
    - 识别高/低置信度区域
    - 可视化不确定性
    - 返回置信度分析
    """

def identify_data_needs(uncertainty_analysis):
    """
    识别需要更多数据的区域
    - 分析不确定性热点
    - 建议数据收集策略
    - 返回数据需求报告
    """
```

---

## 通用工具函数

### 数据处理工具
```python
# 文件：utils.py

def set_random_seeds(student_id):
    """
    根据学生ID设置随机种子
    - 生成5个种子：1XXX, 2XXX, 3XXX, 4XXX, 5XXX
    - 返回种子列表
    """

def check_data_consistency(datasets_dict):
    """
    检查多个数据集的一致性
    - 验证列名是否一致
    - 检查数据类型匹配
    - 识别潜在的数据质量问题
    - 返回一致性检查报告
    """

def stratified_train_test_split(X, y, species_labels, test_size=0.2, random_state=None):
    """
    按分子类型进行分层训练测试集分割
    - 确保每种分子类型在训练和测试集中都有代表
    - 保持各类型的比例分布
    - 返回分层分割后的数据集
    """

def save_results_to_csv(results_dict, file_path):
    """
    保存结果到CSV文件
    - 格式化结果数据
    - 保存到指定路径
    """

def create_visualization(data, plot_type, title, save_path):
    """
    创建和保存可视化图表
    - 支持多种图表类型
    - 保存到figures/目录
    """

def validate_data_quality(df, required_columns):
    """
    验证数据质量
    - 检查必需列
    - 识别异常值
    - 返回质量报告
    """

def error_handling_wrapper(func):
    """
    通用错误处理装饰器
    - 包装函数以处理常见错误
    - 提供有意义的错误消息
    """
```

### 评估工具
```python
def calculate_regression_metrics(y_true, y_pred):
    """
    计算回归评估指标
    - MAE, RMSE, R²
    - 返回指标字典
    """

def calculate_classification_metrics(y_true, y_pred, average='weighted'):
    """
    计算分类评估指标
    - Accuracy, Precision, Recall, F1-score
    - 返回指标字典
    """

def generate_confusion_matrix(y_true, y_pred, labels=None):
    """
    生成和可视化混淆矩阵
    - 计算混淆矩阵
    - 创建热图可视化
    - 返回矩阵和图表
    """
```

---

## 主执行脚本

### 每个问题的主函数
```python
# 在各自的文件中
def main_problem1():
    """
    Problem 1主执行函数
    工作流程：
    1. 加载4个QM9star数据文件
    2. 合并数据集并添加分子类型标识
    3. 数据探索和质量检查
    4. 选择代表性子集和平衡样本
    5. 生成多种分子表示
    6. 训练和评估模型
    7. 分析整体和分子类型特异性性能
    """
    pass

def main_problem2():
    """Problem 2主执行函数"""
    pass

def main_problem3():
    """Problem 3主执行函数"""
    pass

def main_problem4():
    """Problem 4主执行函数"""
    pass
```

---

## 文件结构

```
submission/
├── problem1_molecular_properties.py
├── problem2_reaction_conditions.py  
├── problem3_bayesian_optimization.py
├── problem4_transfer_learning.py
├── utils.py
├── results/
│   ├── problem1_results.csv
│   ├── problem2_results.csv
│   ├── problem3_results.csv
│   └── problem4_results.csv
├── figures/
│   └── [相关图表和可视化]
├── requirements.txt
└── README.md
```

---

## 注意事项

1. **随机种子使用**：每个函数需要支持随机种子参数以确保可重现性
2. **错误处理**：每个函数都应包含适当的错误处理
3. **文档字符串**：所有函数都需要详细的文档字符串
4. **模块化设计**：函数应该是独立的，可重用的
5. **数据验证**：在处理数据前进行质量检查
6. **化学解释**：结果应该包含化学意义的解释
7. **性能监控**：跟踪模型训练和预测时间
8. **内存管理**：处理大型数据集时注意内存使用

这个分解提供了实现所有四个问题所需的完整函数结构。每个函数都有明确的目的和期望的输入/输出，便于逐步实现和测试。
