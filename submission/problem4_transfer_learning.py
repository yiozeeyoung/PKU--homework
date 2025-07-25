#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题4：迁移学习和少样本学习 (Problem 4: Transfer Learning and Few-Shot Learning)

应用迁移学习技术使用来自更大数据集的知识处理有限的AHO数据集

作者：学生ID 153
日期：2025年7月24日
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 机器学习库
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.base import clone

# 分子描述符和化学信息学
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("RDKit not available - using alternative molecular representations")

# 不确定性量化
from scipy import stats
from scipy.stats import norm

# 随机种子设置
STUDENT_ID = 153
RANDOM_SEEDS = [1153, 2153, 3153, 4153, 5153]


def load_datasets():
    """
    加载AHO和OOS数据集
    """
    print("=" * 50)
    print("加载数据集")
    print("=" * 50)
    
    datasets = {}
    
    # 尝试加载AHO数据集
    try:
        aho_df = pd.read_csv("dataset/AHO.csv")
        datasets['AHO'] = aho_df
        print(f"✓ AHO数据集加载成功: {aho_df.shape}")
    except FileNotFoundError:
        print("⚠ AHO.csv未找到，生成模拟数据")
        # 生成模拟AHO数据
        datasets['AHO'] = generate_simulated_data(2000, 'AHO')
    
    # 尝试加载OOS数据集（如果存在）
    try:
        oos_df = pd.read_csv("dataset/AHO_OOS.csv")
        datasets['OOS'] = oos_df
        print(f"✓ OOS数据集加载成功: {oos_df.shape}")
    except FileNotFoundError:
        print("⚠ AHO_OOS.csv未找到，生成模拟数据")
        # 生成模拟OOS数据
        datasets['OOS'] = generate_simulated_data(500, 'OOS')
    
    return datasets


def generate_simulated_data(n_samples, dataset_type):
    """
    生成模拟化学数据
    """
    np.random.seed(RANDOM_SEEDS[0])
    
    # 生成分子特征
    features = {}
    
    # 基本分子性质
    features['molecular_weight'] = np.random.normal(200, 50, n_samples)
    features['logP'] = np.random.normal(2, 1, n_samples)
    features['num_atoms'] = np.random.randint(10, 50, n_samples)
    features['num_bonds'] = features['num_atoms'] + np.random.randint(-5, 15, n_samples)
    features['num_rings'] = np.random.randint(0, 5, n_samples)
    features['aromatic_atoms'] = np.random.randint(0, features['num_atoms'] // 2, n_samples)
    
    # 原子组成
    features['C_count'] = np.random.randint(5, 30, n_samples)
    features['N_count'] = np.random.randint(0, 5, n_samples)
    features['O_count'] = np.random.randint(0, 8, n_samples)
    features['S_count'] = np.random.randint(0, 2, n_samples)
    
    # 键类型统计
    features['single_bonds'] = np.random.randint(5, 25, n_samples)
    features['double_bonds'] = np.random.randint(0, 8, n_samples)
    features['triple_bonds'] = np.random.randint(0, 3, n_samples)
    
    # 官能团
    features['alcohol_groups'] = np.random.randint(0, 3, n_samples)
    features['carbonyl_groups'] = np.random.randint(0, 3, n_samples)
    features['amine_groups'] = np.random.randint(0, 3, n_samples)
    
    df = pd.DataFrame(features)
    
    # 生成目标变量（根据特征的复杂函数）
    if dataset_type == 'AHO':
        # AHO数据集的目标变量
        target = (0.3 * df['molecular_weight'] / 100 + 
                 0.2 * df['logP'] + 
                 0.1 * df['num_rings'] + 
                 0.15 * df['aromatic_atoms'] / 10 +
                 0.1 * df['carbonyl_groups'] +
                 np.random.normal(0, 0.5, n_samples))
    else:
        # OOS数据集的目标变量（略有不同的模式）
        target = (0.25 * df['molecular_weight'] / 100 + 
                 0.25 * df['logP'] + 
                 0.15 * df['num_rings'] + 
                 0.1 * df['aromatic_atoms'] / 10 +
                 0.05 * df['alcohol_groups'] +
                 np.random.normal(0, 0.3, n_samples))
    
    df['target'] = target
    
    print(f"✓ 生成 {dataset_type} 模拟数据: {df.shape}")
    
    return df


def prepare_molecular_features(df):
    """
    准备分子特征
    """
    print(f"准备分子特征...")
    
    # 选择数值特征
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # 排除目标变量
    if 'target' in numeric_features:
        numeric_features.remove('target')
    
    X = df[numeric_features].fillna(0)
    
    # 检查目标变量
    if 'target' in df.columns:
        y = df['target'].values
    else:
        # 如果没有明确的目标变量，尝试找到第一个数值列作为目标
        target_candidates = [col for col in df.columns if 'target' in col.lower() or 'yield' in col.lower()]
        if target_candidates:
            y = df[target_candidates[0]].values
        else:
            # 使用第一个数值列
            y = df.iloc[:, 0].values
    
    print(f"  特征矩阵: {X.shape}")
    print(f"  目标变量: {y.shape}")
    print(f"  特征列: {list(X.columns)}")
    
    return X.values, y


def split_transfer_learning_data(X, y, source_ratio=0.8):
    """
    分割数据以模拟迁移学习场景
    """
    print("=" * 50)
    print("准备迁移学习数据分割")
    print("=" * 50)
    
    # 分割点
    n_samples = len(X)
    split_point = int(source_ratio * n_samples)
    
    # 随机排列
    np.random.seed(RANDOM_SEEDS[0])
    indices = np.random.permutation(n_samples)
    
    # 源域和目标域
    source_indices = indices[:split_point]
    target_indices = indices[split_point:]
    
    X_source = X[source_indices]
    y_source = y[source_indices]
    X_target = X[target_indices]
    y_target = y[target_indices]
    
    print(f"数据分割完成:")
    print(f"  源域: {X_source.shape[0]} 样本")
    print(f"  目标域: {X_target.shape[0]} 样本")
    print(f"  比例: {source_ratio:.1%} : {1-source_ratio:.1%}")
    
    return X_source, y_source, X_target, y_target


def train_baseline_models(X_source, y_source, X_target, y_target):
    """
    训练基准模型
    """
    print("=" * 50)
    print("训练基准模型")
    print("=" * 50)
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEEDS[0]),
        'SVR': SVR(kernel='rbf'),
        'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=RANDOM_SEEDS[0], max_iter=1000)
    }
    
    results = {}
    
    # 在源域训练，在目标域测试
    print("源域训练 -> 目标域测试:")
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    for name, model in models.items():
        try:
            # 训练
            model.fit(X_source_scaled, y_source)
            
            # 在目标域预测
            y_pred = model.predict(X_target_scaled)
            
            # 评估
            mae = mean_absolute_error(y_target, y_pred)
            rmse = np.sqrt(mean_squared_error(y_target, y_pred))
            r2 = r2_score(y_target, y_pred)
            
            results[f'Source_{name}'] = {
                'model': model,
                'scaler': scaler,
                'predictions': y_pred,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"  {name}: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
            
        except Exception as e:
            print(f"  {name} 失败: {e}")
            results[f'Source_{name}'] = None
    
    # 仅在目标域训练和测试（5折交叉验证）
    print("\n仅目标域（交叉验证）:")
    
    for name, model_class in [('RandomForest', RandomForestRegressor), 
                              ('SVR', SVR), 
                              ('MLP', MLPRegressor)]:
        try:
            model = model_class(random_state=RANDOM_SEEDS[0]) if name != 'SVR' else model_class(kernel='rbf')
            if name == 'MLP':
                model.set_params(hidden_layer_sizes=(50,), max_iter=1000)
            
            # 交叉验证
            cv_scores = cross_val_score(model, X_target_scaled, y_target, cv=5, scoring='r2')
            
            # 训练最终模型
            model.fit(X_target_scaled, y_target)
            y_pred = model.predict(X_target_scaled)
            
            mae = mean_absolute_error(y_target, y_pred)
            rmse = np.sqrt(mean_squared_error(y_target, y_pred))
            r2 = r2_score(y_target, y_pred)
            
            results[f'Target_{name}'] = {
                'model': model,
                'scaler': scaler,
                'predictions': y_pred,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"  {name}: CV R²={cv_scores.mean():.3f}±{cv_scores.std():.3f}, R²={r2:.3f}")
            
        except Exception as e:
            print(f"  {name} 失败: {e}")
            results[f'Target_{name}'] = None
    
    return results


def implement_transfer_learning_methods(X_source, y_source, X_target, y_target):
    """
    实现迁移学习方法
    """
    print("=" * 50)
    print("实现迁移学习方法")
    print("=" * 50)
    
    scaler = StandardScaler()
    X_source_scaled = scaler.fit_transform(X_source)
    X_target_scaled = scaler.transform(X_target)
    
    transfer_results = {}
    
    # 1. 特征对齐迁移学习
    print("1. 特征对齐迁移学习...")
    try:
        # 在源域训练模型
        source_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEEDS[0])
        source_model.fit(X_source_scaled, y_source)
        
        # 使用少量目标域数据微调
        if len(X_target) > 10:
            # 选择一小部分目标域数据
            n_adapt = min(len(X_target) // 2, 50)
            adapt_indices = np.random.choice(len(X_target), n_adapt, replace=False)
            
            X_adapt = X_target_scaled[adapt_indices]
            y_adapt = y_target[adapt_indices]
            
            # 微调模型
            adapted_model = RandomForestRegressor(n_estimators=50, random_state=RANDOM_SEEDS[1])
            
            # 混合训练数据
            X_mixed = np.vstack([X_source_scaled, X_adapt])
            y_mixed = np.hstack([y_source, y_adapt])
            
            adapted_model.fit(X_mixed, y_mixed)
            
            # 测试
            test_indices = np.setdiff1d(np.arange(len(X_target)), adapt_indices)
            X_test = X_target_scaled[test_indices]
            y_test = y_target[test_indices]
            
            y_pred = adapted_model.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            transfer_results['Feature_Alignment'] = {
                'model': adapted_model,
                'predictions': y_pred,
                'true_values': y_test,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"   特征对齐: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
        
    except Exception as e:
        print(f"   特征对齐失败: {e}")
        transfer_results['Feature_Alignment'] = None
    
    # 2. 渐进式适应
    print("2. 渐进式适应...")
    try:
        # 使用目标域的一小部分数据逐步适应
        if len(X_target) > 20:
            # 分批适应
            batch_size = max(5, len(X_target) // 4)
            
            current_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEEDS[0])
            current_model.fit(X_source_scaled, y_source)
            
            # 逐步添加目标域数据
            for i in range(0, len(X_target), batch_size):
                end_idx = min(i + batch_size, len(X_target))
                
                X_batch = X_target_scaled[i:end_idx]
                y_batch = y_target[i:end_idx]
                
                # 重新训练模型，包含新的批次
                X_combined = np.vstack([X_source_scaled, X_target_scaled[:end_idx]])
                y_combined = np.hstack([y_source, y_target[:end_idx]])
                
                # 减少源域数据的权重
                sample_weights = np.ones(len(X_combined))
                sample_weights[:len(X_source)] *= 0.5  # 降低源域权重
                
                new_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEEDS[1])
                new_model.fit(X_combined, y_combined, sample_weight=sample_weights)
                current_model = new_model
            
            # 测试最终模型
            y_pred = current_model.predict(X_target_scaled)
            
            mae = mean_absolute_error(y_target, y_pred)
            rmse = np.sqrt(mean_squared_error(y_target, y_pred))
            r2 = r2_score(y_target, y_pred)
            
            transfer_results['Gradual_Adaptation'] = {
                'model': current_model,
                'predictions': y_pred,
                'true_values': y_target,
                'MAE': mae,
                'RMSE': rmse,
                'R2': r2
            }
            
            print(f"   渐进式适应: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
        
    except Exception as e:
        print(f"   渐进式适应失败: {e}")
        transfer_results['Gradual_Adaptation'] = None
    
    # 3. 相似性加权
    print("3. 相似性加权...")
    try:
        # 计算目标域样本与源域样本的相似性
        from sklearn.metrics.pairwise import cosine_similarity
        
        similarities = cosine_similarity(X_target_scaled, X_source_scaled)
        
        # 为每个目标域样本找到最相似的源域样本
        weighted_predictions = []
        
        for i in range(len(X_target)):
            # 获取与目标样本最相似的源域样本
            sim_scores = similarities[i]
            top_k = min(10, len(X_source))  # 使用前10个最相似的样本
            top_indices = np.argsort(sim_scores)[-top_k:]
            
            # 基于相似性加权预测
            X_similar = X_source_scaled[top_indices]
            y_similar = y_source[top_indices]
            weights = sim_scores[top_indices]
            weights = weights / np.sum(weights)  # 归一化权重
            
            # 简单加权平均预测
            weighted_pred = np.average(y_similar, weights=weights)
            weighted_predictions.append(weighted_pred)
        
        y_pred = np.array(weighted_predictions)
        
        mae = mean_absolute_error(y_target, y_pred)
        rmse = np.sqrt(mean_squared_error(y_target, y_pred))
        r2 = r2_score(y_target, y_pred)
        
        transfer_results['Similarity_Weighting'] = {
            'predictions': y_pred,
            'true_values': y_target,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2
        }
        
        print(f"   相似性加权: MAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}")
        
    except Exception as e:
        print(f"   相似性加权失败: {e}")
        transfer_results['Similarity_Weighting'] = None
    
    return transfer_results


def quantify_uncertainty(models_dict, X_target, y_target):
    """
    量化不确定性
    """
    print("=" * 50)
    print("不确定性量化")
    print("=" * 50)
    
    uncertainty_results = {}
    
    # 集成方法的不确定性
    if len([m for m in models_dict.values() if m is not None]) > 1:
        # 收集所有有效模型的预测
        all_predictions = []
        model_names = []
        
        for name, result in models_dict.items():
            if result is not None and 'predictions' in result:
                all_predictions.append(result['predictions'])
                model_names.append(name)
        
        if len(all_predictions) > 1:
            predictions_array = np.array(all_predictions)
            
            # 计算集成统计
            mean_pred = np.mean(predictions_array, axis=0)
            std_pred = np.std(predictions_array, axis=0)
            
            # 置信区间
            confidence_level = 0.95
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            margin_error = z_score * std_pred
            
            lower_bound = mean_pred - margin_error
            upper_bound = mean_pred + margin_error
            
            uncertainty_results = {
                'mean': mean_pred,
                'std': std_pred,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'model_names': model_names,
                'num_models': len(all_predictions)
            }
            
            # 计算置信度分数
            confidence_scores = 1 / (1 + std_pred)  # 标准差越小，置信度越高
            
            # 评估不确定性质量
            if len(mean_pred) > 0:
                mae_ensemble = mean_absolute_error(y_target, mean_pred)
                rmse_ensemble = np.sqrt(mean_squared_error(y_target, mean_pred))
                r2_ensemble = r2_score(y_target, mean_pred)
                
                uncertainty_results.update({
                    'confidence_scores': confidence_scores,
                    'mae': mae_ensemble,
                    'rmse': rmse_ensemble,
                    'r2': r2_ensemble
                })
                
                print(f"集成模型不确定性:")
                print(f"  模型数量: {len(all_predictions)}")
                print(f"  平均不确定性: {np.mean(std_pred):.3f}")
                print(f"  集成性能: MAE={mae_ensemble:.3f}, R²={r2_ensemble:.3f}")
                
                # 分析高低不确定性区域
                high_uncertainty_mask = std_pred > np.percentile(std_pred, 75)
                low_uncertainty_mask = std_pred < np.percentile(std_pred, 25)
                
                if np.any(high_uncertainty_mask) and np.any(low_uncertainty_mask):
                    high_unc_error = np.mean(np.abs(y_target[high_uncertainty_mask] - mean_pred[high_uncertainty_mask]))
                    low_unc_error = np.mean(np.abs(y_target[low_uncertainty_mask] - mean_pred[low_uncertainty_mask]))
                    
                    uncertainty_results.update({
                        'high_uncertainty_error': high_unc_error,
                        'low_uncertainty_error': low_unc_error,
                        'uncertainty_correlation': np.corrcoef(std_pred, np.abs(y_target - mean_pred))[0, 1]
                    })
                    
                    print(f"  高不确定性区域误差: {high_unc_error:.3f}")
                    print(f"  低不确定性区域误差: {low_unc_error:.3f}")
                    print(f"  不确定性-误差相关性: {uncertainty_results['uncertainty_correlation']:.3f}")
    
    return uncertainty_results


def compare_all_methods(baseline_results, transfer_results):
    """
    比较所有方法的性能
    """
    print("=" * 50)
    print("性能比较")
    print("=" * 50)
    
    all_results = {}
    all_results.update(baseline_results)
    all_results.update(transfer_results)
    
    # 整理比较数据
    comparison_data = []
    
    for method_name, result in all_results.items():
        if result is not None:
            comparison_data.append({
                'Method': method_name,
                'Type': 'Baseline' if any(prefix in method_name for prefix in ['Source_', 'Target_']) else 'Transfer',
                'MAE': result.get('MAE', np.nan),
                'RMSE': result.get('RMSE', np.nan),
                'R2': result.get('R2', np.nan),
                'CV_Mean': result.get('cv_mean', np.nan),
                'CV_Std': result.get('cv_std', np.nan)
            })
    
    if comparison_data:
        comparison_df = pd.DataFrame(comparison_data)
        
        print("性能比较:")
        print(comparison_df.round(3))
        
        # 找到最佳方法
        valid_r2 = comparison_df.dropna(subset=['R2'])
        if len(valid_r2) > 0:
            best_method = valid_r2.loc[valid_r2['R2'].idxmax(), 'Method']
            best_r2 = valid_r2['R2'].max()
            print(f"\n最佳方法: {best_method} (R² = {best_r2:.3f})")
        
        return comparison_df
    
    return None


def visualize_results(comparison_df, uncertainty_results, save_dir="submission/figures"):
    """
    可视化结果
    """
    print("=" * 50)
    print("生成可视化")
    print("=" * 50)
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    if comparison_df is not None:
        # 性能比较图
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # R² 比较
        methods = comparison_df['Method']
        r2_values = comparison_df['R2']
        colors = ['blue' if 'Transfer' in t else 'orange' for t in comparison_df['Type']]
        
        axes[0, 0].bar(range(len(methods)), r2_values, color=colors)
        axes[0, 0].set_title('R² Performance Comparison')
        axes[0, 0].set_ylabel('R²')
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels(methods, rotation=45, ha='right')
        
        # MAE 比较
        mae_values = comparison_df['MAE']
        axes[0, 1].bar(range(len(methods)), mae_values, color=colors)
        axes[0, 1].set_title('MAE Performance Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_xticks(range(len(methods)))
        axes[0, 1].set_xticklabels(methods, rotation=45, ha='right')
        
        # 不确定性可视化
        if uncertainty_results and 'std' in uncertainty_results:
            axes[1, 0].hist(uncertainty_results['std'], bins=20, alpha=0.7, color='green')
            axes[1, 0].set_title('Prediction Uncertainty Distribution')
            axes[1, 0].set_xlabel('Standard Deviation')
            axes[1, 0].set_ylabel('Frequency')
        
        # 类型统计
        type_counts = comparison_df['Type'].value_counts()
        axes[1, 1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Method Type Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/problem4_results.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✓ 结果图已保存到: {save_dir}/problem4_results.png")


def save_results(comparison_df, uncertainty_results, output_dir="submission/results"):
    """
    保存结果
    """
    print("=" * 50)
    print("保存结果")
    print("=" * 50)
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存性能比较
    if comparison_df is not None:
        comparison_df.to_csv(f"{output_dir}/problem4_results.csv", index=False)
        print(f"✓ 性能比较已保存到: {output_dir}/problem4_results.csv")
    
    # 保存不确定性结果
    if uncertainty_results:
        uncertainty_summary = {
            'num_models': uncertainty_results.get('num_models', 0),
            'mean_uncertainty': np.mean(uncertainty_results.get('std', [0])),
            'ensemble_mae': uncertainty_results.get('mae', np.nan),
            'ensemble_r2': uncertainty_results.get('r2', np.nan),
            'uncertainty_correlation': uncertainty_results.get('uncertainty_correlation', np.nan)
        }
        
        uncertainty_df = pd.DataFrame([uncertainty_summary])
        uncertainty_df.to_csv(f"{output_dir}/problem4_uncertainty.csv", index=False)
        print(f"✓ 不确定性结果已保存到: {output_dir}/problem4_uncertainty.csv")
    
    # 保存总结
    summary = {
        'student_id': STUDENT_ID,
        'random_seeds': str(RANDOM_SEEDS),
        'best_method': comparison_df.loc[comparison_df['R2'].idxmax(), 'Method'] if comparison_df is not None and not comparison_df['R2'].isna().all() else 'Unknown',
        'transfer_learning_success': 'Yes' if any('Transfer' in comparison_df['Type'].values) else 'No' if comparison_df is not None else 'Unknown'
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{output_dir}/problem4_summary.csv", index=False)
    print(f"✓ 总结已保存到: {output_dir}/problem4_summary.csv")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("问题4：迁移学习和少样本学习")
    print("学生ID: 153")
    print("随机种子:", RANDOM_SEEDS)
    print("=" * 60)
    
    # 1. 加载数据集
    datasets = load_datasets()
    
    # 2. 处理AHO数据集
    aho_data = datasets['AHO']
    X_aho, y_aho = prepare_molecular_features(aho_data)
    
    # 3. 分割数据以模拟迁移学习场景
    X_source, y_source, X_target, y_target = split_transfer_learning_data(X_aho, y_aho)
    
    # 4. 训练基准模型
    baseline_results = train_baseline_models(X_source, y_source, X_target, y_target)
    
    # 5. 实现迁移学习方法
    transfer_results = implement_transfer_learning_methods(X_source, y_source, X_target, y_target)
    
    # 6. 量化不确定性
    all_methods = {}
    all_methods.update({k: v for k, v in baseline_results.items() if v is not None})
    all_methods.update({k: v for k, v in transfer_results.items() if v is not None})
    
    uncertainty_results = quantify_uncertainty(all_methods, X_target, y_target)
    
    # 7. 比较所有方法
    comparison_df = compare_all_methods(baseline_results, transfer_results)
    
    # 8. 可视化结果
    visualize_results(comparison_df, uncertainty_results)
    
    # 9. 保存结果
    save_results(comparison_df, uncertainty_results)
    
    # 10. 提供化学解释
    print("=" * 50)
    print("化学解释和建议")
    print("=" * 50)
    
    print("1. 迁移学习在化学数据中的应用:")
    print("   - 利用大型数据集的知识帮助小数据集的预测")
    print("   - 特别适用于新化合物类别的性质预测")
    print("   - 可以减少实验成本和时间")
    
    print("\n2. 不确定性量化的重要性:")
    print("   - 识别模型预测的可信度")
    print("   - 指导实验设计和数据收集")
    print("   - 避免在不确定区域做出错误决策")
    
    print("\n3. 实际应用建议:")
    print("   - 优先验证高不确定性的预测结果")
    print("   - 在相似化学空间中应用迁移学习")
    print("   - 持续收集数据以改进模型性能")
    
    print("\n=" * 60)
    print("问题4完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
