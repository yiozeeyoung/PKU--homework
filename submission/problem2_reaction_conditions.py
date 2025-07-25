#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
问题2：反应条件预测 (Problem 2: Reaction Condition Prediction)

使用反应数据集A构建多分类模型来预测最优反应条件

作者：学生ID 153
日期：2025年7月24日
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# 机器学习
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputClassifier

# 化学信息学
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 随机种子设置（基于学生ID：153）
STUDENT_ID = 153
RANDOM_SEEDS = [1153, 2153, 3153, 4153, 5153]


def load_reaction_dataset_a(file_path="dataset/Reaction Dataset A.csv"):
    """
    加载反应数据集A
    """
    try:
        df = pd.read_csv(file_path)
        print("=" * 50)
        print("加载反应数据集A")
        print("=" * 50)
        print(f"✓ 成功加载反应数据集A")
        print(f"  数据形状: {df.shape}")
        print(f"  列名: {list(df.columns)}")
        
        # 显示基本信息
        print(f"\n数据集基本信息:")
        print(f"  样本数量: {len(df)}")
        print(f"  特征数量: {len(df.columns)}")
        
        # 检查缺失值
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            print(f"\n缺失值统计:")
            print(missing_values[missing_values > 0])
        else:
            print(f"\n✓ 无缺失值")
            
        return df
        
    except FileNotFoundError:
        print(f"✗ 未找到文件: {file_path}")
        return None
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        return None


def analyze_reaction_conditions(df):
    """
    分析催化剂、碱和溶剂的分布
    """
    print("=" * 50)
    print("反应条件分布分析")
    print("=" * 50)
    
    condition_columns = ['Catalyst', 'Base', 'Solvent']
    analysis_results = {}
    
    for condition in condition_columns:
        if condition in df.columns:
            value_counts = df[condition].value_counts()
            print(f"\n{condition} 分布:")
            print(f"  总类别数: {len(value_counts)}")
            print(f"  前5个类别:")
            for i, (value, count) in enumerate(value_counts.head().items()):
                percentage = (count / len(df)) * 100
                print(f"    {i+1}. {value}: {count} ({percentage:.1f}%)")
            
            analysis_results[condition] = {
                'unique_count': len(value_counts),
                'value_counts': value_counts,
                'most_common': value_counts.index[0],
                'most_common_count': value_counts.iloc[0]
            }
    
    return analysis_results


def create_reaction_fingerprints(df, smiles_columns=None):
    """
    创建反应指纹
    """
    print("=" * 50)
    print("生成反应指纹")
    print("=" * 50)
    
    if smiles_columns is None:
        # 自动检测SMILES列
        smiles_columns = [col for col in df.columns if 'smiles' in col.lower() or 'reactant' in col.lower()]
    
    print(f"检测到SMILES列: {smiles_columns}")
    
    reaction_fingerprints = []
    
    for idx, row in df.iterrows():
        fingerprint = []
        
        # 对每个反应物生成Morgan指纹
        for col in smiles_columns:
            if pd.notna(row[col]):
                mol = Chem.MolFromSmiles(str(row[col]))
                if mol is not None:
                    fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
                    fingerprint.extend(list(fp))
                else:
                    fingerprint.extend([0] * 1024)
            else:
                fingerprint.extend([0] * 1024)
        
        # 如果没有SMILES列，创建空指纹
        if not fingerprint:
            fingerprint = [0] * 2048  # 默认长度
        
        reaction_fingerprints.append(fingerprint)
    
    fingerprint_array = np.array(reaction_fingerprints)
    print(f"✓ 反应指纹生成完成: {fingerprint_array.shape}")
    
    return fingerprint_array


def create_condition_features(df):
    """
    创建条件特征向量
    """
    print("=" * 50)
    print("生成条件特征向量")
    print("=" * 50)
    
    condition_features = []
    encoders = {}
    
    condition_columns = ['Catalyst', 'Base', 'Solvent']
    
    for condition in condition_columns:
        if condition in df.columns:
            # 标签编码
            encoder = LabelEncoder()
            encoded_values = encoder.fit_transform(df[condition].astype(str))
            condition_features.append(encoded_values.reshape(-1, 1))
            encoders[condition] = encoder
            print(f"✓ {condition} 编码完成: {len(encoder.classes_)} 个类别")
        else:
            print(f"⚠ 未找到 {condition} 列")
    
    if condition_features:
        combined_features = np.hstack(condition_features)
        print(f"✓ 条件特征向量生成完成: {combined_features.shape}")
        return combined_features, encoders
    else:
        print("✗ 未找到任何条件列")
        return None, {}


def train_classification_models(X, y_dict, test_size=0.2, cv_folds=5):
    """
    训练多个分类模型
    """
    print("=" * 50)
    print("训练分类模型")
    print("=" * 50)
    
    results = {}
    
    for target_name, y in y_dict.items():
        print(f"\n训练 {target_name} 分类器...")
        
        # 检查类别分布
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"  类别数: {len(unique_classes)}")
        print(f"  样本分布: {dict(zip(unique_classes, counts))}")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=RANDOM_SEEDS[0],
            stratify=y
        )

        # 训练模型
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_SEEDS[0],
            n_jobs=-1
        )

        # 交叉验证仅在训练集上进行
        try:
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=cv_folds, scoring='accuracy',
                n_jobs=-1
            )

            # 训练最终模型
            model.fit(X_train, y_train)

            # 在测试集上预测
            y_pred = model.predict(X_test)

            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            
            results[target_name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'predictions': y_pred,
                'true_values': y_test,
                'unique_classes': unique_classes
            }
            
            print(f"✓ {target_name} 结果:")
            print(f"  交叉验证准确率: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
            print(f"  准确率: {accuracy:.3f}")
            print(f"  精确率: {precision:.3f}")
            print(f"  召回率: {recall:.3f}")
            print(f"  F1分数: {f1:.3f}")
            
        except Exception as e:
            print(f"✗ {target_name} 训练失败: {e}")
            results[target_name] = None
    
    return results


def generate_confusion_matrices(results, save_dir="submission/figures"):
    """
    生成混淆矩阵
    """
    print("=" * 50)
    print("生成混淆矩阵")
    print("=" * 50)
    
    os.makedirs(save_dir, exist_ok=True)
    
    n_models = len([r for r in results.values() if r is not None])
    if n_models == 0:
        print("没有可用的模型结果")
        return
    
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    if n_models == 1:
        axes = [axes]
    
    plot_idx = 0
    
    for target_name, result in results.items():
        if result is None:
            continue
            
        # 计算混淆矩阵
        cm = confusion_matrix(result['true_values'], result['predictions'])
        
        # 绘制混淆矩阵
        ax = axes[plot_idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f'{target_name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        plot_idx += 1
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/problem2_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ 混淆矩阵已保存到: {save_dir}/problem2_confusion_matrices.png")


def analyze_feature_importance(results, feature_names=None):
    """
    分析特征重要性
    """
    print("=" * 50)
    print("特征重要性分析")
    print("=" * 50)
    
    for target_name, result in results.items():
        if result is None or 'model' not in result:
            continue
            
        print(f"\n{target_name} 特征重要性:")
        
        model = result['model']
        importances = model.feature_importances_
        
        # 获取前10个重要特征
        top_indices = np.argsort(importances)[-10:][::-1]
        top_importances = importances[top_indices]
        
        for i, (idx, importance) in enumerate(zip(top_indices, top_importances)):
            feature_name = f"特征_{idx}" if feature_names is None else feature_names[idx]
            print(f"  {i+1}. {feature_name}: {importance:.3f}")


def provide_chemical_interpretation(results, condition_analysis):
    """
    提供化学解释
    """
    print("=" * 50)
    print("化学解释")
    print("=" * 50)
    
    print("1. 反应条件预测的化学意义:")
    print("   - 催化剂选择直接影响反应速率和选择性")
    print("   - 碱的选择影响反应物的去质子化")
    print("   - 溶剂影响反应物溶解性和反应环境")
    
    print("\n2. 模型性能解释:")
    
    for target_name, result in results.items():
        if result is None:
            continue
            
        accuracy = result['accuracy']
        f1 = result['f1_score']
        
        print(f"\n   {target_name} 预测:")
        print(f"   - 准确率: {accuracy:.3f}")
        print(f"   - F1分数: {f1:.3f}")
        
        if accuracy > 0.8:
            print(f"   ✓ 高准确率表明该条件选择具有明确的化学规律")
        elif accuracy > 0.6:
            print(f"   ○ 中等准确率表明该条件选择受多种因素影响")
        else:
            print(f"   ⚠ 较低准确率表明该条件选择可能更依赖经验")
    
    print("\n3. 最佳反应条件组合:")
    for condition, analysis in condition_analysis.items():
        if analysis:
            most_common = analysis['most_common']
            count = analysis['most_common_count']
            print(f"   - 最常用{condition}: {most_common} ({count}次)")
    
    print("\n4. 实际应用建议:")
    print("   - 结合模型预测和化学知识进行条件选择")
    print("   - 对低置信度预测进行实验验证")
    print("   - 考虑反应物结构对条件选择的影响")


def save_results(results, output_dir="submission/results"):
    """
    保存结果
    """
    print("=" * 50)
    print("保存结果")
    print("=" * 50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 整理结果数据
    results_data = []
    
    for target_name, result in results.items():
        if result is not None:
            results_data.append({
                'Target': target_name,
                'CV_Accuracy_Mean': result['cv_mean'],
                'CV_Accuracy_Std': result['cv_std'],
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1_Score': result['f1_score'],
                'Num_Classes': len(result['unique_classes'])
            })
    
    # 保存为CSV
    if results_data:
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(f"{output_dir}/problem2_results.csv", index=False)
        print(f"✓ 结果已保存到: {output_dir}/problem2_results.csv")
        
        # 显示结果摘要
        print(f"\n结果摘要:")
        print(results_df.round(3))
    else:
        print("✗ 没有可保存的结果")


def main():
    """
    主函数
    """
    print("=" * 60)
    print("问题2：反应条件预测")
    print("学生ID: 153")
    print("随机种子:", RANDOM_SEEDS)
    print("=" * 60)
    
    # 1. 加载数据
    df = load_reaction_dataset_a()
    if df is None:
        print("数据加载失败")
        return
    
    # 2. 分析反应条件分布
    condition_analysis = analyze_reaction_conditions(df)
    
    # 3. 生成反应指纹
    reaction_fps = create_reaction_fingerprints(df)
    
    # 4. 生成条件特征向量
    condition_features, encoders = create_condition_features(df)
    
    # 5. 合并特征
    if condition_features is not None:
        X = np.hstack([reaction_fps, condition_features])
        print(f"✓ 总特征维度: {X.shape}")
    else:
        X = reaction_fps
        print(f"✓ 使用反应指纹特征: {X.shape}")
    
    # 6. 准备目标变量
    target_dict = {}
    for condition in ['Catalyst', 'Base', 'Solvent']:
        if condition in df.columns:
            if condition in encoders:
                target_dict[condition] = encoders[condition].transform(df[condition].astype(str))
            else:
                encoder = LabelEncoder()
                target_dict[condition] = encoder.fit_transform(df[condition].astype(str))
    
    if not target_dict:
        print("✗ 未找到有效的目标变量")
        return
    
    # 7. 训练模型
    results = train_classification_models(X, target_dict)
    
    # 8. 生成混淆矩阵
    generate_confusion_matrices(results)
    
    # 9. 分析特征重要性
    analyze_feature_importance(results)
    
    # 10. 提供化学解释
    provide_chemical_interpretation(results, condition_analysis)
    
    # 11. 保存结果
    save_results(results)
    
    print("\n=" * 60)
    print("问题2完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
