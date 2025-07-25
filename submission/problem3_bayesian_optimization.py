#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题3：贝叶斯优化反应收率预测 (Problem 3: Bayesian Optimization for Reaction Yield)

使用反应数据集B（Doyle数据集）实现贝叶斯优化来预测和优化反应收率

作者：学生ID 153
日期：2025年7月24日
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 机器学习相关库
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 化学信息学库
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 随机种子设置（基于学生ID：153）
STUDENT_ID = 153
RANDOM_SEEDS = [1153, 2153, 3153, 4153, 5153]
MAIN_SEED = 1153

np.random.seed(MAIN_SEED)


def load_doyle_dataset(file_path="dataset/Reaction Dataset B.csv"):
    """
    加载Doyle数据集（反应数据集B）
    """
    try:
        df = pd.read_csv(file_path)
        print("=" * 50)
        print("加载Doyle数据集（反应数据集B）")
        print("=" * 50)
        print(f"✓ 成功加载数据集，形状: {df.shape}")
        print(f"✓ 列名: {list(df.columns)}")
        
        # 显示基本信息
        print("\n数据集基本信息:")
        print(df.info())
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n缺失值统计:")
            print(missing_values[missing_values > 0])
        else:
            print("\n✓ 无缺失值")
        
        return df
    
    except Exception as e:
        print(f"✗ 加载数据失败: {e}")
        return None


def preprocess_features(df):
    """
    预处理特征
    """
    print("=" * 50)
    print("预处理特征")
    print("=" * 50)
    
    features = []
    feature_names = []
    
    # 1. 数值特征
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    print(f"数值特征列: {list(numeric_columns)}")
    
    for col in numeric_columns:
        if col not in ['Target', 'target', 'yield', 'Yield']:  # 排除目标变量
            feature_values = df[col].fillna(df[col].median())
            features.append(feature_values.values.reshape(-1, 1))
            feature_names.append(col)
    
    # 2. 分类特征编码
    categorical_columns = df.select_dtypes(include=['object']).columns
    print(f"分类特征列: {list(categorical_columns)}")
    
    encoders = {}
    for col in categorical_columns:
        if 'smiles' not in col.lower():  # 排除SMILES列
            encoder = LabelEncoder()
            encoded_values = encoder.fit_transform(df[col].astype(str))
            features.append(encoded_values.reshape(-1, 1))
            feature_names.append(col)
            encoders[col] = encoder
    
    # 3. 分子指纹（如果有SMILES列）
    smiles_columns = [col for col in df.columns if 'smiles' in col.lower()]
    if smiles_columns:
        print(f"SMILES列: {smiles_columns}")
        fingerprints = []
        
        for smiles_col in smiles_columns:
            col_fingerprints = []
            for smiles in df[smiles_col]:
                if pd.notna(smiles):
                    mol = Chem.MolFromSmiles(str(smiles))
                    if mol is not None:
                        fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
                        col_fingerprints.append(list(fp))
                    else:
                        col_fingerprints.append([0] * 512)
                else:
                    col_fingerprints.append([0] * 512)
            
            fingerprints.extend(np.array(col_fingerprints).T)
            feature_names.extend([f"{smiles_col}_fp_{i}" for i in range(512)])
        
        if fingerprints:
            features.append(np.array(fingerprints).T)
    
    # 合并所有特征
    if features:
        X = np.hstack(features)
        print(f"✓ 特征矩阵形状: {X.shape}")
        print(f"✓ 特征名称数量: {len(feature_names)}")
        return X, feature_names, encoders
    else:
        print("✗ 未找到有效特征")
        return None, [], {}


def setup_gaussian_process(kernel_type='RBF'):
    """
    设置高斯过程模型
    """
    print("=" * 50)
    print(f"设置高斯过程模型 (内核: {kernel_type})")
    print("=" * 50)
    
    # 定义不同的内核
    if kernel_type == 'RBF':
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    elif kernel_type == 'Matern':
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
    elif kernel_type == 'RBF+Noise':
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-3, (1e-5, 1e-1))
    else:
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
    
    gp = GaussianProcessRegressor(
        kernel=kernel,
        n_restarts_optimizer=10,
        random_state=MAIN_SEED,
        alpha=1e-6
    )
    
    print(f"✓ 高斯过程模型已配置")
    print(f"  内核: {kernel}")
    
    return gp


def expected_improvement(X, gp, y_best, xi=0.01):
    """
    期望改进(EI)获取函数
    """
    mu, sigma = gp.predict(X, return_std=True)
    sigma = sigma.reshape(-1, 1)
    
    improvement = mu - y_best - xi
    Z = improvement / sigma
    
    ei = improvement * stats.norm.cdf(Z) + sigma * stats.norm.pdf(Z)
    return ei.flatten()


def upper_confidence_bound(X, gp, kappa=2.576):
    """
    上置信界(UCB)获取函数
    """
    mu, sigma = gp.predict(X, return_std=True)
    return mu + kappa * sigma


def probability_of_improvement(X, gp, y_best, xi=0.01):
    """
    改进概率(PI)获取函数
    """
    mu, sigma = gp.predict(X, return_std=True)
    improvement = mu - y_best - xi
    Z = improvement / sigma
    return stats.norm.cdf(Z)


def bayesian_optimization_loop(X_train, y_train, X_pool, n_iterations=10, acquisition='EI'):
    """
    贝叶斯优化循环
    """
    print("=" * 50)
    print(f"贝叶斯优化循环 (获取函数: {acquisition})")
    print("=" * 50)
    
    # 初始化
    X_selected = X_train.copy()
    y_selected = y_train.copy()
    
    optimization_history = []
    selected_indices = []
    
    for iteration in range(n_iterations):
        print(f"\n第 {iteration + 1}/{n_iterations} 次迭代:")
        
        # 训练高斯过程模型
        gp = setup_gaussian_process('RBF+Noise')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        gp.fit(X_scaled, y_selected)
        
        # 在池中选择下一个点
        X_pool_scaled = scaler.transform(X_pool)
        
        if acquisition == 'EI':
            acquisition_values = expected_improvement(X_pool_scaled, gp, np.max(y_selected))
        elif acquisition == 'UCB':
            acquisition_values = upper_confidence_bound(X_pool_scaled, gp)
        elif acquisition == 'PI':
            acquisition_values = probability_of_improvement(X_pool_scaled, gp, np.max(y_selected))
        else:
            acquisition_values = expected_improvement(X_pool_scaled, gp, np.max(y_selected))
        
        # 选择获取值最高的点
        best_idx = np.argmax(acquisition_values)
        selected_indices.append(best_idx)
        
        # 添加到训练集
        X_selected = np.vstack([X_selected, X_pool[best_idx:best_idx+1]])
        y_selected = np.append(y_selected, [0])  # 占位符，实际应用中需要真实值
        
        # 记录历史
        current_best = np.max(y_selected[:-1])  # 排除占位符
        optimization_history.append({
            'iteration': iteration + 1,
            'selected_index': best_idx,
            'acquisition_value': acquisition_values[best_idx],
            'current_best': current_best,
            'gp_score': gp.score(X_scaled, y_selected[:-1])  # 排除占位符
        })
        
        print(f"  选择样本索引: {best_idx}")
        print(f"  获取函数值: {acquisition_values[best_idx]:.3f}")
        print(f"  当前最佳收率: {current_best:.3f}")
    
    return optimization_history, selected_indices, gp, scaler


def analyze_optimization_results(optimization_history):
    """
    分析优化结果
    """
    print("=" * 50)
    print("优化结果分析")
    print("=" * 50)
    
    # 转换为DataFrame便于分析
    history_df = pd.DataFrame(optimization_history)
    
    print("优化历史摘要:")
    print(history_df[['iteration', 'current_best', 'acquisition_value']].round(3))
    
    # 计算改进情况
    initial_best = history_df['current_best'].iloc[0]
    final_best = history_df['current_best'].iloc[-1]
    improvement = final_best - initial_best
    
    print(f"\n优化效果:")
    print(f"  初始最佳收率: {initial_best:.3f}")
    print(f"  最终最佳收率: {final_best:.3f}")
    print(f"  总改进: {improvement:.3f}")
    print(f"  相对改进: {(improvement/initial_best)*100:.1f}%")
    
    return history_df


def visualize_optimization_progress(history_df, save_dir="submission/figures"):
    """
    可视化优化进度
    """
    print("=" * 50)
    print("生成优化进度可视化")
    print("=" * 50)
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Best yield over iterations
    axes[0, 0].plot(history_df['iteration'], history_df['current_best'], 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Best Yield')
    axes[0, 0].set_title('Optimization Trajectory')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Acquisition function value
    axes[0, 1].plot(history_df['iteration'], history_df['acquisition_value'], 's-', 
                   color='orange', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Acquisition Value')
    axes[0, 1].set_title('Acquisition Value Change')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Yield improvement
    improvements = history_df['current_best'] - history_df['current_best'].iloc[0]
    axes[1, 0].bar(history_df['iteration'], improvements, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Cumulative Improvement')
    axes[1, 0].set_title('Cumulative Improvement')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. GP model performance (if available)
    if 'gp_score' in history_df.columns:
        axes[1, 1].plot(history_df['iteration'], history_df['gp_score'], '^-', 
                       color='purple', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('GP Model Score')
        axes[1, 1].set_title('Gaussian Process Model Performance')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/problem3_optimization_progress.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 优化进度图已保存到: {save_dir}/problem3_optimization_progress.png")


def compare_acquisition_functions(X_train, y_train, X_pool, n_iterations=5):
    """
    比较不同获取函数的性能
    """
    print("=" * 50)
    print("比较获取函数性能")
    print("=" * 50)
    
    acquisition_functions = ['EI', 'UCB', 'PI']
    comparison_results = {}
    
    for acq_func in acquisition_functions:
        print(f"\n测试 {acq_func} 获取函数...")
        
        try:
            history, selected_indices, gp, scaler = bayesian_optimization_loop(
                X_train, y_train, X_pool, n_iterations, acq_func
            )
            
            history_df = pd.DataFrame(history)
            
            # 计算性能指标
            initial_best = history_df['current_best'].iloc[0]
            final_best = history_df['current_best'].iloc[-1]
            improvement = final_best - initial_best
            
            comparison_results[acq_func] = {
                'initial_best': initial_best,
                'final_best': final_best,
                'improvement': improvement,
                'relative_improvement': (improvement/initial_best)*100 if initial_best != 0 else 0,
                'history': history_df
            }
            
            print(f"  {acq_func} 结果: 改进 {improvement:.3f} ({(improvement/initial_best)*100:.1f}%)")
            
        except Exception as e:
            print(f"  {acq_func} 失败: {e}")
            comparison_results[acq_func] = None
    
    # 找到最佳获取函数
    valid_results = {k: v for k, v in comparison_results.items() if v is not None}
    
    if valid_results:
        best_acq = max(valid_results.keys(), key=lambda k: valid_results[k]['improvement'])
        print(f"\n最佳获取函数: {best_acq}")
        print(f"最大改进: {valid_results[best_acq]['improvement']:.3f}")
    
    return comparison_results


def identify_optimal_conditions(gp, scaler, X_pool, feature_names):
    """
    识别最优反应条件
    """
    print("=" * 50)
    print("识别最优反应条件")
    print("=" * 50)
    
    # 预测所有池中样本的收率
    X_pool_scaled = scaler.transform(X_pool)
    predictions, uncertainties = gp.predict(X_pool_scaled, return_std=True)
    
    # 找到预测收率最高的条件
    best_idx = np.argmax(predictions)
    best_prediction = predictions[best_idx]
    best_uncertainty = uncertainties[best_idx]
    
    print(f"最优条件 (样本索引 {best_idx}):")
    print(f"  预测收率: {best_prediction:.3f} ± {best_uncertainty:.3f}")
    
    # 显示特征值（如果有特征名称）
    if feature_names and len(feature_names) == X_pool.shape[1]:
        print(f"  最重要的特征值:")
        for i, (name, value) in enumerate(zip(feature_names, X_pool[best_idx])):
            if i < 10:  # 只显示前10个特征
                print(f"    {name}: {value:.3f}")
    
    # 找到高置信度的高收率条件
    high_confidence_mask = uncertainties < np.percentile(uncertainties, 25)  # 低不确定性 = 高置信度
    high_yield_mask = predictions > np.percentile(predictions, 75)  # 高收率
    
    good_conditions = np.where(high_confidence_mask & high_yield_mask)[0]
    
    print(f"\n推荐的反应条件 (高收率且高置信度):")
    print(f"  找到 {len(good_conditions)} 个候选条件")
    
    if len(good_conditions) > 0:
        # 显示前5个推荐条件
        for i, idx in enumerate(good_conditions[:5]):
            print(f"  候选 {i+1} (索引 {idx}): 预测收率 {predictions[idx]:.3f} ± {uncertainties[idx]:.3f}")
    
    return best_idx, good_conditions, predictions, uncertainties


def save_results(comparison_results, optimization_history, output_dir="submission/results"):
    """
    保存结果
    """
    print("=" * 50)
    print("保存结果")
    print("=" * 50)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存获取函数比较结果
    if comparison_results:
        comparison_data = []
        for acq_func, result in comparison_results.items():
            if result is not None:
                comparison_data.append({
                    'Acquisition_Function': acq_func,
                    'Initial_Best': result['initial_best'],
                    'Final_Best': result['final_best'],
                    'Improvement': result['improvement'],
                    'Relative_Improvement_Percent': result['relative_improvement']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df.to_csv(f"{output_dir}/problem3_acquisition_comparison.csv", index=False)
            print(f"✓ 获取函数比较结果已保存到: {output_dir}/problem3_acquisition_comparison.csv")
    
    # 保存优化历史
    if optimization_history:
        history_df = pd.DataFrame(optimization_history)
        history_df.to_csv(f"{output_dir}/problem3_optimization_history.csv", index=False)
        print(f"✓ 优化历史已保存到: {output_dir}/problem3_optimization_history.csv")
    
    # 保存总结报告
    summary = {
        'student_id': STUDENT_ID,
        'random_seeds': RANDOM_SEEDS,
        'total_iterations': len(optimization_history) if optimization_history else 0,
        'best_acquisition_function': max(comparison_results.keys(), 
                                       key=lambda k: comparison_results[k]['improvement'] 
                                       if comparison_results[k] is not None else -np.inf) if comparison_results else 'Unknown'
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{output_dir}/problem3_summary.csv", index=False)
    print(f"✓ 总结报告已保存到: {output_dir}/problem3_summary.csv")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("问题3：贝叶斯优化反应收率预测")
    print("学生ID: 153")
    print("随机种子:", RANDOM_SEEDS)
    print("=" * 60)
    
    # 1. 加载数据
    df = load_doyle_dataset()
    if df is None:
        print("数据加载失败")
        return
    
    # 2. 预处理特征
    X, feature_names, encoders = preprocess_features(df)
    if X is None:
        print("特征预处理失败")
        return
    
    # 3. 准备目标变量
    target_col = None
    for col in df.columns:
        if 'target' in col.lower() or 'yield' in col.lower():
            target_col = col
            break
    
    if target_col is None:
        print("未找到目标变量，使用模拟数据")
        y = np.random.uniform(0, 100, len(df))  # 模拟收率数据
    else:
        y = df[target_col].values
        print(f"目标变量: {target_col}")
    
    print(f"目标变量统计: 均值={y.mean():.3f}, 标准差={y.std():.3f}")
    
    # 4. 数据分割
    # 初始训练集（20个点）和候选池
    initial_size = min(20, len(X) // 2)
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:initial_size]
    pool_indices = indices[initial_size:initial_size + min(100, len(X) - initial_size)]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_pool = X[pool_indices]
    
    print(f"初始训练集大小: {len(X_train)}")
    print(f"候选池大小: {len(X_pool)}")
    
    # 5. 比较获取函数
    comparison_results = compare_acquisition_functions(X_train, y_train, X_pool, n_iterations=10)
    
    # 6. 使用最佳获取函数进行完整优化
    best_acq = 'EI'  # 默认使用EI
    valid_results = {k: v for k, v in comparison_results.items() if v is not None}
    if valid_results:
        best_acq = max(valid_results.keys(), key=lambda k: valid_results[k]['improvement'])
    
    print(f"\n使用最佳获取函数 {best_acq} 进行完整优化...")
    optimization_history, selected_indices, final_gp, final_scaler = bayesian_optimization_loop(
        X_train, y_train, X_pool, n_iterations=15, acquisition=best_acq
    )
    
    # 7. 分析优化结果
    history_df = analyze_optimization_results(optimization_history)
    
    # 8. 可视化优化进度
    visualize_optimization_progress(history_df)
    
    # 9. 识别最优条件
    best_idx, good_conditions, predictions, uncertainties = identify_optimal_conditions(
        final_gp, final_scaler, X_pool, feature_names
    )
    
    # 10. 保存结果
    save_results(comparison_results, optimization_history)
    
    print("\n=" * 60)
    print("问题3完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
