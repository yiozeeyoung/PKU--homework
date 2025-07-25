#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题1：分子性质预测 (Problem 1: Molecular Property Prediction)

使用QM9star数据集开发和比较多种分子表示方法来预测HOMO-LUMO能隙

作者：学生ID 153
日期：2025年7月24日
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 机器学习
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 化学信息学
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import rdBase
rdBase.DisableLog('rdApp.warning')

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 随机种子设置（基于学生ID：153）
STUDENT_ID = 153
RANDOM_SEEDS = [1153, 2153, 3153, 4153, 5153]


def load_qm9star_datasets(dataset_dir="dataset"):
    """
    加载QM9star数据集的所有4个文件
    """
    datasets = {}
    file_mappings = {
        'anion': 'qm9star_anion_FMO.csv',
        'cation': 'qm9star_cation_FMO.csv',
        'neutral': 'qm9star_neutral_FMO.csv',
        'radical': 'qm9star_radical_FMO.csv'
    }
    
    print("=== 加载QM9star数据集 ===")
    
    for species_type, filename in file_mappings.items():
        file_path = os.path.join(dataset_dir, filename)
        
        try:
            df = pd.read_csv(file_path)
            datasets[species_type] = df
            print(f"✓ 成功加载 {filename}: {df.shape[0]} 行, {df.shape[1]} 列")
        except FileNotFoundError:
            print(f"✗ 未找到文件: {file_path}")
        except Exception as e:
            print(f"✗ 加载 {filename} 时出错: {str(e)}")
    
    return datasets


def merge_datasets(datasets_dict):
    """
    合并4个QM9star数据集
    """
    if not datasets_dict:
        print("错误：数据集字典为空")
        return None
    
    merged_dataframes = []
    
    print("=== 合并数据集 ===")
    
    for species_type, df in datasets_dict.items():
        df_copy = df.copy()
        df_copy['molecular_species'] = species_type
        merged_dataframes.append(df_copy)
        print(f"添加 {species_type}: {len(df_copy)} 行")
    
    merged_df = pd.concat(merged_dataframes, ignore_index=True)
    print(f"✓ 合并完成: {merged_df.shape[0]} 行, {merged_df.shape[1]} 列")
    
    return merged_df


def clean_smiles_data(merged_df):
    """
    清理SMILES数据
    """
    print("=== 清理SMILES数据 ===")
    
    initial_count = len(merged_df)
    
    # 检查SMILES列
    smiles_col = None
    for col in merged_df.columns:
        if 'smiles' in col.lower() and col != 'canonical_smiles':
            smiles_col = col
            break
    
    if smiles_col is None:
        print("未找到SMILES列")
        return merged_df
    
    # 移除空值
    merged_df = merged_df.dropna(subset=[smiles_col])
    
    # 验证SMILES有效性
    valid_smiles = []
    for idx, smiles in enumerate(merged_df[smiles_col]):
        mol = Chem.MolFromSmiles(smiles)
        valid_smiles.append(mol is not None)
    
    merged_df = merged_df[valid_smiles].reset_index(drop=True)
    
    final_count = len(merged_df)
    removed_count = initial_count - final_count
    
    print(f"✓ 数据清理完成:")
    print(f"  原始数据: {initial_count} 行")
    print(f"  清理后: {final_count} 行")
    print(f"  移除: {removed_count} 行 ({removed_count/initial_count*100:.2f}%)")
    
    return merged_df


def sample_data_by_species(merged_df, n_samples=100000):
    """
    从每个分子类型中抽样代表性子集
    """
    print(f"=== 数据抽样 (每类型 {n_samples:,} 样本) ===")
    
    sampled_dataframes = []
    
    for species in merged_df['molecular_species'].unique():
        species_df = merged_df[merged_df['molecular_species'] == species]
        
        if len(species_df) > n_samples:
            sampled_df = species_df.sample(n=n_samples, random_state=RANDOM_SEEDS[0])
            print(f"{species}: 从 {len(species_df):,} 中抽样 {n_samples:,}")
        else:
            sampled_df = species_df
            print(f"{species}: 使用全部 {len(species_df):,} 样本")
        
        sampled_dataframes.append(sampled_df)
    
    final_df = pd.concat(sampled_dataframes, ignore_index=True)
    print(f"✓ 抽样完成: {len(final_df):,} 总样本")
    
    return final_df


def generate_molecular_representations(df, smiles_col='smiles'):
    """
    生成三种分子表示方法
    """
    print("=== 生成分子表示 ===")
    
    representations = {}
    
    # 1. Morgan指纹
    print("1. 生成Morgan指纹...")
    morgan_fps = []
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            morgan_fps.append(list(fp))
        else:
            morgan_fps.append([0] * 2048)
    
    representations['Morgan指纹'] = np.array(morgan_fps)
    print(f"   ✓ Morgan指纹: {representations['Morgan指纹'].shape}")
    
    # 2. RDKit描述符
    print("2. 计算RDKit描述符...")
    descriptor_names = [x[0] for x in Descriptors._descList]
    descriptor_calc = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    
    rdkit_descriptors = []
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            descriptors = list(descriptor_calc.CalcDescriptors(mol))
            # 处理NaN值
            descriptors = [0 if pd.isna(d) or np.isinf(d) else d for d in descriptors]
            rdkit_descriptors.append(descriptors)
        else:
            rdkit_descriptors.append([0] * len(descriptor_names))
    
    representations['RDKit描述符'] = np.array(rdkit_descriptors)
    print(f"   ✓ RDKit描述符: {representations['RDKit描述符'].shape}")
    
    # 3. 原子中心特征
    print("3. 生成原子中心特征...")
    atom_features = []
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            features = extract_atom_centered_features(mol)
            atom_features.append(features)
        else:
            atom_features.append([0] * 50)  # 默认特征长度
    
    representations['原子中心特征'] = np.array(atom_features)
    print(f"   ✓ 原子中心特征: {representations['原子中心特征'].shape}")
    
    return representations


def extract_atom_centered_features(mol):
    """
    提取原子中心特征
    """
    if mol is None:
        return [0] * 50
    
    features = []
    
    # 原子类型统计
    atom_counts = {'C': 0, 'N': 0, 'O': 0, 'S': 0, 'P': 0, 'F': 0, 'Cl': 0, 'Br': 0, 'I': 0}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in atom_counts:
            atom_counts[symbol] += 1
    
    features.extend(list(atom_counts.values()))  # 9个特征
    
    # 分子性质
    features.append(mol.GetNumAtoms())  # 原子数
    features.append(mol.GetNumBonds())  # 键数
    features.append(rdMolDescriptors.CalcNumRings(mol))  # 环数
    features.append(rdMolDescriptors.CalcNumAromaticRings(mol))  # 芳香环数
    features.append(rdMolDescriptors.CalcNumHeteroatoms(mol))  # 杂原子数
    features.append(rdMolDescriptors.CalcNumRotatableBonds(mol))  # 可旋转键数
    features.append(rdMolDescriptors.CalcNumHBD(mol))  # 氢键供体数
    features.append(rdMolDescriptors.CalcNumHBA(mol))  # 氢键受体数
    
    # 杂化状态统计
    hybridization_counts = [0, 0, 0, 0]  # sp, sp2, sp3, other
    for atom in mol.GetAtoms():
        hyb = atom.GetHybridization()
        if hyb == Chem.HybridizationType.SP:
            hybridization_counts[0] += 1
        elif hyb == Chem.HybridizationType.SP2:
            hybridization_counts[1] += 1
        elif hyb == Chem.HybridizationType.SP3:
            hybridization_counts[2] += 1
        else:
            hybridization_counts[3] += 1
    
    features.extend(hybridization_counts)  # 4个特征
    
    # 芳香性原子统计
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    features.append(aromatic_atoms)
    
    # 电荷统计
    formal_charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
    features.append(sum(formal_charges))  # 总电荷
    features.append(sum(abs(charge) for charge in formal_charges))  # 绝对电荷和
    
    # 填充到固定长度
    while len(features) < 50:
        features.append(0)
    
    return features[:50]


def train_and_evaluate_models(representations, target_values, cv_folds=5):
    """
    训练和评估随机森林模型
    """
    print("=== 模型训练和评估 ===")
    
    results = {}
    
    for method_name, X in representations.items():
        print(f"\n训练 {method_name} 模型...")
        
        # 数据预处理
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 训练模型
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_SEEDS[0],
            n_jobs=-1
        )
        
        print(f"特征矩阵形状: {X_scaled.shape}")
        print(f"目标变量形状: {target_values.shape}")
        
        # 交叉验证
        cv_scores = cross_val_score(
            model, X_scaled, target_values,
            cv=cv_folds, scoring='r2',
            n_jobs=-1
        )
        
        # 训练最终模型
        model.fit(X_scaled, target_values)
        
        # 预测
        y_pred = model.predict(X_scaled)
        
        # 计算指标
        mae = mean_absolute_error(target_values, y_pred)
        rmse = np.sqrt(mean_squared_error(target_values, y_pred))
        r2 = r2_score(target_values, y_pred)
        
        results[method_name] = {
            'model': model,
            'scaler': scaler,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2,
            'predictions': y_pred
        }
        
        print(f"✓ {method_name} 结果:")
        print(f"  交叉验证 R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  MAE: {mae:.3f}")
        print(f"  RMSE: {rmse:.3f}")
        print(f"  R²: {r2:.3f}")
    
    return results


def analyze_feature_importance(results, method_name='RDKit描述符'):
    """
    分析特征重要性
    """
    print(f"=== {method_name} 特征重要性分析 ===")
    
    if method_name not in results:
        print(f"未找到 {method_name} 的结果")
        return
    
    model = results[method_name]['model']
    importances = model.feature_importances_
    
    # 获取前10个重要特征
    top_indices = np.argsort(importances)[-10:][::-1]
    top_importances = importances[top_indices]
    
    print("前10个重要特征:")
    for i, (idx, importance) in enumerate(zip(top_indices, top_importances)):
        print(f"  {i+1}. 特征 {idx}: {importance:.3f}")


def compare_model_performance(results):
    """
    比较模型性能
    """
    print("=== 模型性能比较 ===")
    
    comparison_df = pd.DataFrame({
        method: {
            'CV R²': result['cv_mean'],
            'CV Std': result['cv_std'],
            'MAE': result['MAE'],
            'RMSE': result['RMSE'],
            'R²': result['R²']
        }
        for method, result in results.items()
    }).T
    
    print(comparison_df.round(3))
    
    # 找到最佳方法
    best_method = comparison_df['R²'].idxmax()
    print(f"\n最佳方法: {best_method} (R² = {comparison_df.loc[best_method, 'R²']:.3f})")
    
    return comparison_df


def save_results(results, comparison_df, output_dir="submission/results"):
    """
    保存结果
    """
    print("=== 保存结果 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存性能比较
    comparison_df.to_csv(f"{output_dir}/problem1_results.csv")
    print(f"✓ 性能比较已保存到: {output_dir}/problem1_results.csv")
    
    # 保存详细结果
    detailed_results = []
    for method, result in results.items():
        detailed_results.append({
            'Method': method,
            'CV_R2_Mean': result['cv_mean'],
            'CV_R2_Std': result['cv_std'],
            'MAE': result['MAE'],
            'RMSE': result['RMSE'],
            'R2': result['R²']
        })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(f"{output_dir}/problem1_detailed_results.csv", index=False)
    print(f"✓ 详细结果已保存到: {output_dir}/problem1_detailed_results.csv")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("问题1：分子性质预测")
    print("学生ID: 153")
    print("随机种子:", RANDOM_SEEDS)
    print("=" * 60)
    
    # 1. 加载数据
    datasets = load_qm9star_datasets()
    if not datasets:
        print("数据加载失败")
        return
    
    # 2. 合并数据
    merged_df = merge_datasets(datasets)
    if merged_df is None:
        print("数据合并失败")
        return
    
    # 3. 清理数据
    cleaned_df = clean_smiles_data(merged_df)
    
    # 4. 抽样数据
    sampled_df = sample_data_by_species(cleaned_df, n_samples=100000)
    
    # 5. 准备目标变量
    target_col = None
    for col in sampled_df.columns:
        if 'gap' in col.lower() or ('homo' in col.lower() and 'lumo' in col.lower()):
            target_col = col
            break
    
    if target_col is None:
        print("未找到HOMO-LUMO能隙列")
        return
    
    y = sampled_df[target_col].values
    print(f"目标变量: {target_col}")
    print(f"目标变量统计: 均值={y.mean():.3f}, 标准差={y.std():.3f}")
    
    # 6. 生成分子表示
    representations = generate_molecular_representations(sampled_df)
    
    # 7. 训练和评估模型
    results = train_and_evaluate_models(representations, y)
    
    # 8. 分析特征重要性
    analyze_feature_importance(results)
    
    # 9. 比较模型性能
    comparison_df = compare_model_performance(results)
    
    # 10. 保存结果
    save_results(results, comparison_df)
    
    print("\n=" * 60)
    print("问题1完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
