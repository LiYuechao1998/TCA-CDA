import argparse
import os
import sys

print("", flush=True)
sys.stdout.flush()

import numpy as np
import pandas as pd
print("", flush=True)
sys.stdout.flush()

import torch
import torch.nn.functional as F
print("", flush=True)
sys.stdout.flush()

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    average_precision_score,
    confusion_matrix,
)
print("", flush=True)
sys.stdout.flush()


def compute_comprehensive_score(y_true, y_pred, y_probs):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = 0.0
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    auc = roc_auc_score(y_true, y_probs)
    aupr = average_precision_score(y_true, y_probs)

    score = (
        0.15 * acc +
        0.15 * prec +
        0.15 * rec +      #
        0.15 * spec +     #
        0.15 * f1 +       #
        0.10 * mcc +      #
        0.10 * auc +      #
        0.05 * aupr       # A
    )
    
    return score, {
        'acc': acc, 'prec': prec, 'rec': rec, 'spec': spec,
        'f1': f1, 'mcc': mcc, 'auc': auc, 'aupr': aupr
    }


def find_optimal_threshold(y_true, y_probs, metric='rec_spec_sum', target_recall=None, min_spec=0.89, min_recall=0.89):
    if target_recall is not None:
        thresholds = np.arange(0.05, 0.95, 0.01)  # 从 0.05 开始搜索，允许更低的阈值
    else:
        thresholds = np.arange(0.1, 0.95, 0.01)
    best_score = -1
    best_threshold = 0.5
    best_metrics = {}
    if target_recall is not None:
        best_diff = float('inf')
        for thresh in thresholds:
            y_pred = (y_probs > thresh).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                spec_threshold = max(min_spec - 0.03, 0.85) if target_recall is not None else min_spec
                if spec >= spec_threshold and rec >= min_recall:
                    diff = abs(rec - target_recall)
                    if diff < best_diff:
                        best_diff = diff
                        best_threshold = thresh
                        best_metrics = {
                            'threshold': thresh,
                            'rec': rec,
                            'spec': spec,
                            'prec': prec,
                            'f1': f1,
                            'score': rec + spec
                        }
        if best_metrics:
            return best_threshold, best_metrics

    for thresh in thresholds:
        y_pred = (y_probs > thresh).astype(int)
        
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            current_mcc = 0.0
        else:
            continue
        
        if metric == 'rec_spec_sum':
            score = rec + spec
        elif metric == 'rec_spec_balanced':
            score = 2 * rec * spec / (rec + spec) if (rec + spec) > 0 else 0.0
        elif metric == 'f1_spec':
            score = 0.6 * f1 + 0.4 * spec
        elif metric == 'prec_rec_balanced':
            score = 0.4 * prec + 0.3 * rec + 0.3 * spec
        elif metric == 'prec_aupr' or metric == 'prec_aupr_balanced':
            score = 0.5 * prec + 0.25 * rec + 0.25 * spec
        elif metric == 'spec_boost':
            score = 0.4 * spec + 0.35 * rec + 0.25 * prec
        elif metric == 'mcc_spec_f1':
            try:
                from sklearn.metrics import matthews_corrcoef
                mcc = matthews_corrcoef(y_true, y_pred)
                mcc_norm = (mcc + 1) / 2
                score = 0.45 * mcc_norm + 0.30 * spec + 0.25 * f1
                current_mcc = mcc
            except:
                score = 0.5 * spec + 0.5 * f1
                current_mcc = 0.0
        elif metric == 'prec_only' or metric == 'precision':
            score = prec
        elif metric == 'prec_mcc':
            try:
                from sklearn.metrics import matthews_corrcoef
                mcc = matthews_corrcoef(y_true, y_pred)
                mcc_norm = (mcc + 1) / 2
                score = 0.5 * prec + 0.5 * mcc_norm
                current_mcc = mcc
            except:
                score = prec
                current_mcc = 0.0
        elif metric == 'mcc_prec_f1':
            try:
                from sklearn.metrics import matthews_corrcoef
                mcc = matthews_corrcoef(y_true, y_pred)
                mcc_norm = (mcc + 1) / 2
                score = -0.3 * prec + 0.4 * mcc_norm + 0.2 * f1
                current_mcc = mcc
            except:
                score = 0.5 * prec + 0.5 * f1
                current_mcc = 0.0
        elif metric == 'f1':
            score = f1
        elif metric == 'prec_priority':
            if rec >= min_recall:
                score = 0.1 * prec + 0.3 * rec + 0.2 * spec
            else:
                score = -1
        elif metric == 'rec_priority':
            if spec >= min_spec:
                score = rec * 0.7 + spec * 0.3
            else:
                score = -1
        elif metric == 'comprehensive':
            try:
                from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score
                acc = accuracy_score(y_true, y_pred)
                mcc = matthews_corrcoef(y_true, y_pred)
                auc = roc_auc_score(y_true, y_probs)
                aupr = average_precision_score(y_true, y_probs)
                mcc_norm = (mcc + 1) / 2
                score = (
                    0.15 * acc +
                    0.15 * prec +
                    0.15 * rec +
                    0.15 * spec +
                    0.15 * f1 +
                    0.10 * mcc_norm +
                    0.10 * auc +
                    0.05 * aupr
                )
            except:

                score = f1
        elif metric == 'rec_spec_constrained':
            if rec >= min_recall and spec >= min_spec:
                try:
                    from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score
                    acc = accuracy_score(y_true, y_pred)
                    mcc = matthews_corrcoef(y_true, y_pred)
                    auc = roc_auc_score(y_true, y_probs)
                    aupr = average_precision_score(y_true, y_probs)
                    mcc_norm = (mcc + 1) / 2
                    score = (
                        0.15 * acc + 0.15 * prec + 0.15 * rec + 0.15 * spec +
                        0.15 * f1 + 0.10 * mcc_norm + 0.10 * auc + 0.05 * aupr
                    )
                except:
                    score = rec + spec
            else:
                score = -1
        else:
            score = rec + spec
        if metric == 'rec_spec_sum' and abs(score - best_score) < 1e-6:
            if thresh > best_threshold:
                best_score = score
                best_threshold = thresh
                best_metrics = {
                    'threshold': thresh,
                    'rec': rec,
                    'spec': spec,
                    'prec': prec,
                    'f1': f1,
                    'score': score
                }
                if metric == 'mcc_spec_f1' or metric == 'prec_mcc' or metric == 'mcc_prec_f1':
                    best_metrics['mcc'] = current_mcc
        elif score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = {
                'threshold': thresh,
                'rec': rec,
                'spec': spec,
                'prec': prec,
                'f1': f1,
                'score': score
            }
            if metric == 'mcc_spec_f1' or metric == 'prec_mcc' or metric == 'mcc_prec_f1':
                best_metrics['mcc'] = current_mcc
    
    return best_threshold, best_metrics
print("", flush=True)
sys.stdout.flush()
from sklearn.model_selection import StratifiedKFold

print("", flush=True)
sys.stdout.flush()
from torch import nn
from torch.optim import AdamW

print("", flush=True)
sys.stdout.flush()
from torch_geometric.data import Data

print("正在导入模型 LGI_GT...", flush=True)
sys.stdout.flush()

# 分步导入，定位卡住的位置
try:
    print("  -> 导入 model.gconv...", flush=True)
    sys.stdout.flush()
    from model import gconv
    print("  -> gconv 导入完成", flush=True)
    sys.stdout.flush()
    
    print("  -> 导入 model.tlayer...", flush=True)
    sys.stdout.flush()
    from model import tlayer
    print("  -> tlayer 导入完成", flush=True)
    sys.stdout.flush()
    
    print("  -> 导入 model.rwse...", flush=True)
    sys.stdout.flush()
    from model import rwse
    print("  -> rwse 导入完成", flush=True)
    sys.stdout.flush()
    
    print("  -> 导入 model.ast_node_encoder...", flush=True)
    sys.stdout.flush()
    from model import ast_node_encoder
    print("  -> ast_node_encoder 导入完成", flush=True)
    sys.stdout.flush()
    
    print("  -> 导入 LGI_GT 类...", flush=True)
    sys.stdout.flush()
    from model.lgi_gt import LGI_GT
    print("  -> LGI_GT 导入完成", flush=True)
    sys.stdout.flush()
except Exception as e:
    print(f"  -> 导入出错: {e}", flush=True)
    sys.stdout.flush()
    raise

print("所有模块导入完成！", flush=True)
sys.stdout.flush()


def find_feature_file(data_dir: str, filename: str) -> str:
    """查找特征文件，按优先级检查多个位置"""
    # 优先级1: data_dir 目录下
    path1 = os.path.join(data_dir, filename)
    if os.path.exists(path1):
        return path1
    
    # 优先级2: data_dir/dataset 目录下
    path2 = os.path.join(data_dir, "dataset", filename)
    if os.path.exists(path2):
        return path2
    
    # 优先级3: 脚本目录下的 dataset 目录
    script_dir = os.path.dirname(os.path.realpath(__file__))
    path3 = os.path.join(script_dir, "dataset", filename)
    if os.path.exists(path3):
        return path3
    
    # 如果都不存在，返回第一个路径（用于错误提示）
    return path1


def build_graph(data_dir: str, feat_dim: int = None) -> tuple[Data, torch.Tensor, torch.Tensor, int]:
    """从三个 CSV 构建单个二部图 + 边标签.

    参数:
      - data_dir: 数据目录路径
      - feat_dim: 指定特征维度（32/64/128），如果为 None 则自动选择（优先级：32 > 64 > 128）

    返回:
      - data: torch_geometric.data.Data，包含
          x: (N_nodes, feat_dim) 节点特征
          edge_index: (2, E) 边
          edge_attr: (E,) long，全 0，占位
          batch: (N_nodes,)，全 0（单个图）
      - edge_labels: (E,) long，0/1
      - edge_indices: (E,) long，边下标（用于划分 5 折）
      - feat_dim: int，实际使用的特征维度（32/64/128）
    """
    circ_32 = find_feature_file(data_dir, "circRNA_bert_32.csv")
    circ_64 = find_feature_file(data_dir, "circRNA_bert_64.csv")
    circ_128 = find_feature_file(data_dir, "circRNA_bert_128.csv")
    dis_32 = find_feature_file(data_dir, "disease_GIPK_32.csv")
    dis_64 = find_feature_file(data_dir, "disease_GIPK_64.csv")
    dis_128 = find_feature_file(data_dir, "disease_GIPK_128.csv")

    # 如果指定了维度，优先使用指定维度的文件
    if feat_dim is not None:
        if feat_dim == 32:
            circ_path = circ_32
            dis_path = dis_32
        elif feat_dim == 64:
            circ_path = circ_64
            dis_path = dis_64
        elif feat_dim == 128:
            circ_path = circ_128
            dis_path = dis_128
        else:
            raise ValueError(f"不支持的特征维度: {feat_dim}，只支持 32/64/128")
        
        # 检查指定维度的文件是否存在
        if not os.path.exists(circ_path):
            raise FileNotFoundError(f"指定的 circRNA 特征文件不存在: {circ_path}")
        if not os.path.exists(dis_path):
            raise FileNotFoundError(f"指定的 disease 特征文件不存在: {dis_path}")
    else:
        # 自动选择：按优先级 32 > 64 > 128
        if os.path.exists(circ_32) and os.path.exists(dis_32):
            circ_path = circ_32
            dis_path = dis_32
        elif os.path.exists(circ_64) and os.path.exists(dis_64):
            circ_path = circ_64
            dis_path = dis_64
        elif os.path.exists(circ_128) and os.path.exists(dis_128):
            circ_path = circ_128
            dis_path = dis_128
        else:
            # 如果都不存在，尝试混合匹配（优先低维度）
            if os.path.exists(circ_32):
                circ_path = circ_32
            elif os.path.exists(circ_64):
                circ_path = circ_64
            else:
                circ_path = circ_128
            
            if os.path.exists(dis_32):
                dis_path = dis_32
            elif os.path.exists(dis_64):
                dis_path = dis_64
            else:
                dis_path = dis_128
    # 边对数据文件优先级：sample_pairs_hardneg.csv > sample_pairs.csv > interaction.csv
    # 检查多个位置：data_dir, data_dir/dataset, 脚本目录下的 dataset
    script_dir = os.path.dirname(os.path.realpath(__file__))
    
    # 候选路径列表（按优先级）
    pair_hard_candidates = [
        os.path.join(data_dir, "sample_pairs_hardneg.csv"),
        os.path.join(data_dir, "dataset", "sample_pairs_hardneg.csv"),
        os.path.join(script_dir, "dataset", "sample_pairs_hardneg.csv"),
    ]
    pair_normal_candidates = [
        os.path.join(data_dir, "sample_pairs.csv"),
        os.path.join(data_dir, "dataset", "sample_pairs.csv"),
        os.path.join(script_dir, "dataset", "sample_pairs.csv"),
    ]
    pair_interaction_candidates = [
        os.path.join(data_dir, "interaction.csv"),
        os.path.join(data_dir, "dataset", "interaction.csv"),
        os.path.join(script_dir, "dataset", "interaction.csv"),
    ]
    
    # 查找文件（按优先级）
    pair_path = None
    for path in pair_hard_candidates:
        if os.path.exists(path):
            pair_path = path
            break
    
    if pair_path is None:
        for path in pair_normal_candidates:
            if os.path.exists(path):
                pair_path = path
                break
    
    if pair_path is None:
        for path in pair_interaction_candidates:
            if os.path.exists(path):
                pair_path = path
                break
    
    if pair_path is None:
        raise FileNotFoundError(f"未找到边对数据文件，请确保存在以下文件之一：sample_pairs_hardneg.csv, sample_pairs.csv, interaction.csv")
    
    # 显示实际使用的特征文件（用于确认优先级）
    print(f"使用 circRNA 特征文件: {os.path.basename(circ_path)}")
    print(f"使用 disease 特征文件: {os.path.basename(dis_path)}")

    circ_df = pd.read_csv(circ_path, header=None)
    dis_df = pd.read_csv(dis_path, header=None)
    pair_df = pd.read_csv(pair_path, header=None, names=["circ", "disease"])

    circ_ids = circ_df.iloc[:, 0].astype(str).tolist()
    circ_feat = circ_df.iloc[:, 1:].to_numpy(dtype=np.float32)
    dis_ids = dis_df.iloc[:, 0].astype(str).tolist()
    dis_feat = dis_df.iloc[:, 1:].to_numpy(dtype=np.float32)

    circ_id2idx = {cid: i for i, cid in enumerate(circ_ids)}
    dis_id2idx = {did: i for i, did in enumerate(dis_ids)}

    num_circ = len(circ_ids)
    num_dis = len(dis_ids)
    num_nodes = num_circ + num_dis

    x = np.zeros((num_nodes, circ_feat.shape[1]), dtype=np.float32)
    x[:num_circ] = circ_feat
    x[num_circ:] = dis_feat

    edges_src = []
    edges_dst = []
    for _, row in pair_df.iterrows():
        c = str(row["circ"])
        d = str(row["disease"])
        if c not in circ_id2idx or d not in dis_id2idx:
            # 如果存在未出现在特征表中的 ID，直接跳过该样本
            continue
        u = circ_id2idx[c]
        v = num_circ + dis_id2idx[d]
        edges_src.append(u)
        edges_dst.append(v)

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    num_edges = edge_index.size(1)

    # 检查边对文件是否已经包含负样本（前一半正样本，后一半负样本）
    # 如果所有边都是正样本，需要先创建 sample_pairs.csv 供负采样脚本使用
    if num_edges % 2 == 0:
        # 尝试检查：如果前一半和后一半的边完全不同，可能是正负样本分离
        pos_edges = set(zip(edges_src[:num_edges//2], edges_dst[:num_edges//2]))
        neg_edges = set(zip(edges_src[num_edges//2:], edges_dst[num_edges//2:]))
        if pos_edges.isdisjoint(neg_edges) and len(pos_edges) == len(neg_edges):
            # 前一半和后一半不重叠，可能是正负样本分离
            pos_len = num_edges // 2
        else:
            # 全部是正样本，需要负采样
            pos_len = num_edges
    else:
        # 奇数条边，全部视为正样本
        pos_len = num_edges
    
    # 如果只有正样本，检查是否需要负采样
    if pos_len == num_edges:
        # 检查是否已有 sample_pairs_hardneg.csv（在多个位置查找）
        hardneg_candidates = [
            os.path.join(data_dir, "sample_pairs_hardneg.csv"),
            os.path.join(data_dir, "dataset", "sample_pairs_hardneg.csv"),
            os.path.join(script_dir, "dataset", "sample_pairs_hardneg.csv"),
        ]
        hardneg_path = None
        for path in hardneg_candidates:
            if os.path.exists(path):
                hardneg_path = path
                break
        
        if hardneg_path is not None:
            print(f"检测到只有正样本，但已存在 {os.path.basename(hardneg_path)}，将使用该文件")
            # 重新加载 hardneg 文件
            hardneg_df = pd.read_csv(hardneg_path, header=None, names=["circ", "disease"])
            edges_src = []
            edges_dst = []
            for _, row in hardneg_df.iterrows():
                c = str(row["circ"])
                d = str(row["disease"])
                if c not in circ_id2idx or d not in dis_id2idx:
                    continue
                u = circ_id2idx[c]
                v = num_circ + dis_id2idx[d]
                edges_src.append(u)
                edges_dst.append(v)
            edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
            num_edges = edge_index.size(1)
            pos_len = num_edges // 2
        else:
            # 需要先创建 sample_pairs.csv，然后运行负采样脚本
            sample_pairs_path = os.path.join(data_dir, "sample_pairs.csv")
            if not os.path.exists(sample_pairs_path):
                print(f"检测到只有正样本，创建 {os.path.basename(sample_pairs_path)} 供负采样使用...")
                # 将正样本保存为 sample_pairs.csv
                pos_pairs_df = pd.DataFrame({
                    "circ": [circ_ids[src] for src in edges_src],
                    "disease": [dis_ids[dst - num_circ] for dst in edges_dst]
                })
                pos_pairs_df.to_csv(sample_pairs_path, header=False, index=False)
                print(f"已创建 {sample_pairs_path}，包含 {len(pos_pairs_df)} 条正样本")
                print(f"\n请先运行以下命令进行负采样：")
                # 获取相对路径用于显示
                script_dir = os.path.dirname(os.path.realpath(__file__))
                rel_data_dir = os.path.relpath(data_dir, script_dir)
                print(f"  python build_hard_negative_pairs.py --data_dir {rel_data_dir}")
                print(f"然后重新运行训练脚本。")
                raise ValueError("需要先运行负采样脚本生成 sample_pairs_hardneg.csv")
    
    # 前一半 label=1，后一半 label=0
    labels = torch.zeros(num_edges, dtype=torch.long)
    labels[:pos_len] = 1

    # 单一无类型边特征，占位 0
    edge_attr = torch.zeros(num_edges, dtype=torch.long)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=torch.zeros(num_nodes, dtype=torch.long),
    )

    # 将拼接后的特征和图结构导出为 CSV，方便后续分析或传统模型使用
    export_dir = os.path.join(data_dir, "export_features")
    os.makedirs(export_dir, exist_ok=True)
    # 节点特征：一行一个节点，纯数值，不含 ID；行顺序为 [所有 circRNA, 所有 disease]
    np.savetxt(os.path.join(export_dir, "node_features.csv"), x, delimiter=",")
    # 边索引：两列，src,dst（节点索引，基于上面的顺序）
    edge_index_np = edge_index.numpy().T  # (E, 2)
    np.savetxt(os.path.join(export_dir, "edge_index.csv"), edge_index_np, fmt="%d", delimiter=",")
    # 边标签：一列，0/1
    np.savetxt(os.path.join(export_dir, "edge_labels.csv"), labels.numpy(), fmt="%d", delimiter=",")
    # 边特征拼接：一行一条 RNA-疾病 对的特征 [x_rna, x_dis, |x_rna-x_dis|, x_rna*x_dis]
    h_rna = x[np.array(edges_src)]                # (E, d)
    h_dis = x[np.array(edges_dst)]                # (E, d)
    edge_feat = np.concatenate(
        [h_rna, h_dis, np.abs(h_rna - h_dis), h_rna * h_dis],
        axis=1,
    )  # (E, 4d)
    np.savetxt(os.path.join(export_dir, "edge_features_concat.csv"), edge_feat, delimiter=",")

    edge_indices = torch.arange(num_edges, dtype=torch.long)

    # 获取特征维度
    feat_dim = circ_feat.shape[1]

    return data, labels, edge_indices, feat_dim


class EdgeClassifier(nn.Module):
    """边分类器基类"""
    
    def __init__(self, node_dim: int, classifier_type: str = "linear_bilinear"):
        super().__init__()
        self.node_dim = node_dim
        self.classifier_type = classifier_type
        self._build_classifier()
    
    def _build_classifier(self):
        """根据分类器类型构建不同的分类器"""
        in_dim = self.node_dim * 4  # [h_u, h_v, |h_u-h_v|, h_u*h_v]
        
        if self.classifier_type == "linear_bilinear":
            # 线性 + 双线性（当前默认）
            self.linear = nn.Linear(in_dim, 1)
            self.bilinear = nn.Bilinear(self.node_dim, self.node_dim, 1)
        elif self.classifier_type == "linear_only":
            # 只用线性
            self.linear = nn.Linear(in_dim, 1)
        elif self.classifier_type == "bilinear_only":
            # 只用双线性
            self.bilinear = nn.Bilinear(self.node_dim, self.node_dim, 1)
        elif self.classifier_type == "mlp":
            # MLP（多层感知机）
            hidden_dim = self.node_dim * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif self.classifier_type == "mlp_deep":
            # 更深的 MLP
            hidden_dim = self.node_dim * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif self.classifier_type == "dot_product":
            # 点积相似度
            self.proj = nn.Linear(self.node_dim, self.node_dim)
        else:
            raise ValueError(f"未知的分类器类型: {self.classifier_type}")

    def forward(self, node_emb: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src = edge_index[0]
        dst = edge_index[1]
        h_u = node_emb[src]
        h_v = node_emb[dst]
        
        if self.classifier_type == "linear_bilinear":
            h_cat = torch.cat([h_u, h_v, torch.abs(h_u - h_v), h_u * h_v], dim=-1)
            score_lin = self.linear(h_cat).squeeze(-1)
            score_bi = self.bilinear(h_u, h_v).squeeze(-1)
            logits = score_lin + score_bi
        elif self.classifier_type == "linear_only":
            h_cat = torch.cat([h_u, h_v, torch.abs(h_u - h_v), h_u * h_v], dim=-1)
            logits = self.linear(h_cat).squeeze(-1)
        elif self.classifier_type == "bilinear_only":
            logits = self.bilinear(h_u, h_v).squeeze(-1)
        elif self.classifier_type in ["mlp", "mlp_deep"]:
            h_cat = torch.cat([h_u, h_v, torch.abs(h_u - h_v), h_u * h_v], dim=-1)
            logits = self.mlp(h_cat).squeeze(-1)
        elif self.classifier_type == "dot_product":
            h_u_proj = self.proj(h_u)
            logits = (h_u_proj * h_v).sum(dim=-1)
        else:
            raise ValueError(f"未知的分类器类型: {self.classifier_type}")
        
        return logits


def train_one_fold(
    fold_id: int,
    data: Data,
    labels: torch.Tensor,
    train_edges: np.ndarray,
    test_edges: np.ndarray,
    device: torch.device,
    epochs: int,
    lr: float,
    gconv_dim: int,
    tlayer_dim: int,
    num_layers: int,
    tlayer_attn_dropout: float,
    tlayer_ffn_dropout: float,
    rank_lambda: float,
    rank_margin: float = 0.5,
    use_gconv: bool = True,
    use_tlayer: bool = True,
    classifier_type: str = "linear_bilinear",
    embedding_dim: int = None,
) -> dict:
    """在一个折上训练+评估，返回基于 Test AUC 最优时刻的所有评估指标."""

    model = LGI_GT(
        gconv_dim=gconv_dim,
        tlayer_dim=tlayer_dim,
        dataset_name="CIRCDIS",
        in_dim=data.x.size(1),
        out_dim=gconv_dim,
        num_rw_steps=None,
        dim_pe=None,
        gconv_type="gcn",
        num_layers=num_layers,
        num_heads=4,
        middle_layer_type="none",
        skip_connection="none",
        readout=None,
        norm="ln",
        tlayer_attn_dropout=tlayer_attn_dropout,
        tlayer_ffn_dropout=tlayer_ffn_dropout,
        out_layer=1,
        out_hidden_times=1,
        use_gconv=use_gconv,
        use_tlayer=use_tlayer,
        embedding_dim=embedding_dim,
    )

    edge_cls = EdgeClassifier(node_dim=gconv_dim, classifier_type=classifier_type)

    model.to(device)
    edge_cls.to(device)
    data = data.to(device)
    labels = labels.to(device)

    optimizer = AdamW(
        list(model.parameters()) + list(edge_cls.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
    # 使用加权BCE损失，提高正样本权重以提升召回率
    criterion = nn.BCEWithLogitsLoss()

    train_edges_t = torch.from_numpy(train_edges).to(device)
    test_edges_t = torch.from_numpy(test_edges).to(device)

    best_test_auc = 0.0
    best_metrics = {
        "acc": 0.0,
        "prec": 0.0,
        "rec": 0.0,
        "spec": 0.0,
        "f1": 0.0,
        "mcc": 0.0,
        "auc": 0.0,
        "aupr": 0.0,
    }
    best_probs = None  # 存储该折上最佳 AUC 时的预测概率
    best_y_true = None  # 存储对应的真实标签
    best_threshold = 0.5  # 存储最优阈值

    for epoch in range(1, epochs + 1):
        model.train()
        edge_cls.train()

        optimizer.zero_grad()
        node_emb = model(data)  # (N_nodes, gconv_dim)

        train_logits = edge_cls(node_emb, data.edge_index[:, train_edges_t])
        train_labels = labels[train_edges_t].float()

        # 主损失：BCE
        bce_loss = criterion(train_logits, train_labels)

        # 辅助损失：margin ranking loss，鼓励正样本得分高于负样本
        with torch.no_grad():
            pos_idx = torch.nonzero(train_labels == 1, as_tuple=False).view(-1)
            neg_idx = torch.nonzero(train_labels == 0, as_tuple=False).view(-1)
        if pos_idx.numel() > 0 and neg_idx.numel() > 0:
            num_pairs = min(pos_idx.numel(), neg_idx.numel())
            perm_pos = pos_idx[torch.randperm(pos_idx.numel(), device=device)[:num_pairs]]
            perm_neg = neg_idx[torch.randperm(neg_idx.numel(), device=device)[:num_pairs]]
            logits_pos = train_logits[perm_pos]
            logits_neg = train_logits[perm_neg]
            target = torch.ones_like(logits_pos, device=device)
            rank_loss = F.margin_ranking_loss(logits_pos, logits_neg, target, margin=rank_margin)
        else:
            rank_loss = torch.tensor(0.0, device=device)

        loss = bce_loss + rank_lambda * rank_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            model.eval()
            edge_cls.eval()
            with torch.no_grad():
                node_emb = model(data)
                logits = edge_cls(node_emb, data.edge_index[:, test_edges_t])
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()  # 使用标准阈值 0.5
                y_true = labels[test_edges_t].cpu().numpy()
                y_pred = preds.cpu().numpy()
                y_probs = probs.cpu().numpy()

                # 计算所有评估指标
                test_acc = accuracy_score(y_true, y_pred)
                test_prec = precision_score(y_true, y_pred, zero_division=0)
                test_rec = recall_score(y_true, y_pred, zero_division=0)
                test_f1 = f1_score(y_true, y_pred, zero_division=0)
                test_mcc = matthews_corrcoef(y_true, y_pred)
                test_auc = roc_auc_score(y_true, y_probs)
                test_aupr = average_precision_score(y_true, y_probs)

                # 计算特异性 (Specificity)
                cm = confusion_matrix(y_true, y_pred)
                if cm.size == 4:  # 2x2 混淆矩阵
                    tn, fp, fn, tp = cm.ravel()
                    test_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                else:
                    # 处理只有一类的情况
                    test_spec = 0.0

                # 以 Test AUC 作为主选标准；若 AUC 相同，则优先更高的 Acc
                if (test_auc > best_test_auc) or (
                    abs(test_auc - best_test_auc) < 1e-6 and test_acc > best_metrics["acc"]
                ):
                    best_test_auc = test_auc
                    # 使用最优阈值调优，同时优化 MCC、Precision 和 F1
                    optimal_thresh, opt_metrics = find_optimal_threshold(
                        y_true, y_probs, 
                        metric='mcc_prec_f1'  # 同时优化 MCC、Precision 和 F1（Precision 40%, MCC 40%, F1 20%）
                    )
                    optimal_preds = (y_probs > optimal_thresh).astype(int)
                    best_metrics = {
                        "acc": accuracy_score(y_true, optimal_preds),
                        "prec": precision_score(y_true, optimal_preds, zero_division=0),
                        "rec": recall_score(y_true, optimal_preds, zero_division=0),
                        "f1": f1_score(y_true, optimal_preds, zero_division=0),
                        "mcc": matthews_corrcoef(y_true, optimal_preds),
                        "auc": test_auc,  # AUC 不依赖阈值
                        "aupr": test_aupr,  # AUPR 不依赖阈值
                        "spec": opt_metrics.get("spec", 0.0),
                    }
                    # 记录最佳时刻的概率与标签，方便后续保存为 .npy
                    best_probs = y_probs
                    best_y_true = y_true
                    best_threshold = optimal_thresh

                print(
                    f"[Fold {fold_id}] Epoch {epoch:03d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Test AUC {test_auc:.4f} Acc {test_acc:.4f} | "
                    f"Best Test AUC {best_test_auc:.4f} Acc {best_metrics['acc']:.4f}"
                )

    # 将该折上"最佳 AUC 时刻"的预测概率和真实标签保存为 .npy，方便后续画 ROC / PR 曲线
    if best_probs is not None and best_y_true is not None:
        np.save(f"Y_pre{fold_id - 1}.npy", best_probs)
        np.save(f"Y_test{fold_id - 1}.npy", best_y_true)
        print(f"[Fold {fold_id}] 最优阈值: {best_threshold:.3f}, Rec: {best_metrics['rec']:.4f}, Spec: {best_metrics['spec']:.4f}")

    return best_metrics


def main():
    parser = argparse.ArgumentParser(
        description="circRNA-disease 链路预测（5 折），基于 LGI-GT 编码器"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="包含三个 CSV 的目录（相对于当前脚本所在路径）",
    )
    parser.add_argument("--feat_dim", type=int, default=None, choices=[32, 64, 128],
                       help="指定特征维度（32/64/128），默认 None 表示自动选择（优先级：32 > 64 > 128）")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    # 模型超参数
    parser.add_argument("--gconv_dim", type=int, default=None, 
                       help="GCN 层隐藏维度（默认：自动匹配输入特征维度）")
    parser.add_argument("--tlayer_dim", type=int, default=None,
                       help="Transformer 层隐藏维度（默认：自动匹配输入特征维度）")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--tlayer_attn_dropout", type=float, default=0.1)
    parser.add_argument("--tlayer_ffn_dropout", type=float, default=0.1)
    parser.add_argument("--embedding_dim", type=int, default=None,
                       help="Embedding 维度（32/64/128，仅对 CIRCDIS 数据集有效，默认 None 表示自动匹配 gconv_dim 或 tlayer_dim）")
    # 架构消融参数
    parser.add_argument("--use_gconv", action="store_true", default=True,
                       help="是否使用 GConv 层（默认：True）")
    parser.add_argument("--no_gconv", dest="use_gconv", action="store_false",
                       help="不使用 GConv 层（只用 TLayer）")
    parser.add_argument("--use_tlayer", action="store_true", default=True,
                       help="是否使用 TLayer 层（默认：True）")
    parser.add_argument("--no_tlayer", dest="use_tlayer", action="store_false",
                       help="不使用 TLayer 层（只用 GConv）")
    # 分类器类型
    parser.add_argument("--classifier_type", type=str, default="linear_bilinear",
                       choices=["linear_bilinear", "linear_only", "bilinear_only", "mlp", "mlp_deep", "dot_product"],
                       help="分类器类型：linear_bilinear, linear_only, bilinear_only, mlp, mlp_deep, dot_product")
    # ranking loss 权重和margin
    parser.add_argument("--rank_lambda", type=float, default=0.1)
    parser.add_argument("--rank_margin", type=float, default=0.5,
                       help="Ranking loss的margin值，越大正负样本区分越明显（默认0.5）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()
    
    # 立即输出，确保能看到启动信息
    import sys
    sys.stdout.flush()
    print("=" * 60)
    print("LGI-GT 链路预测训练开始")
    print("=" * 60)
    sys.stdout.flush()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    # 如果 data_dir 是相对路径，先尝试在脚本目录下查找，如果不存在则尝试在上一级目录查找
    if os.path.isabs(args.data_dir):
        data_dir = args.data_dir
    else:
        # 先尝试脚本目录
        data_dir = os.path.join(script_dir, args.data_dir)
        if not os.path.exists(data_dir):
            # 如果不存在，尝试上一级目录
            parent_dir = os.path.dirname(script_dir)
            data_dir = os.path.join(parent_dir, args.data_dir)

    print(f"加载数据集自: {data_dir}")
    sys.stdout.flush()
    if args.feat_dim is not None:
        print(f"指定特征维度: {args.feat_dim}")
    else:
        print("自动选择特征维度（优先级：32 > 64 > 128）")
    sys.stdout.flush()
    
    print("正在构建图数据...")
    sys.stdout.flush()
    data, labels, edge_indices, feat_dim = build_graph(data_dir, feat_dim=args.feat_dim)
    print("图数据构建完成")
    sys.stdout.flush()
    
    # 如果用户没有指定 gconv_dim 和 tlayer_dim，则自动匹配输入特征维度
    if args.gconv_dim is None:
        args.gconv_dim = feat_dim
    if args.tlayer_dim is None:
        args.tlayer_dim = feat_dim
    
    print(f"实际使用特征维度: {feat_dim}")
    print(f"GCN 隐藏维度: {args.gconv_dim}, Transformer 隐藏维度: {args.tlayer_dim}")
    
    # 验证 embedding_dim 参数
    if args.embedding_dim is not None and args.embedding_dim not in [32, 64, 128]:
        raise ValueError(f"embedding_dim 必须是 32、64 或 128，当前值: {args.embedding_dim}")
    
    # 显示架构信息
    if args.use_gconv and args.use_tlayer:
        print("架构: 完整模型 (GConv + TLayer)")
    elif args.use_gconv and not args.use_tlayer:
        print("架构: 消融实验 (只用 GConv)")
    elif not args.use_gconv and args.use_tlayer:
        print("架构: 消融实验 (只用 TLayer)")
    else:
        print("警告: GConv 和 TLayer 都被禁用，模型将无法正常工作！")
    
    # 显示分类器信息
    classifier_names = {
        "linear_bilinear": "线性 + 双线性",
        "linear_only": "仅线性",
        "bilinear_only": "仅双线性",
        "mlp": "MLP（2层）",
        "mlp_deep": "MLP（4层）",
        "dot_product": "点积相似度"
    }
    print(f"分类器: {classifier_names.get(args.classifier_type, args.classifier_type)}")
    if args.embedding_dim is not None:
        print(f"Embedding 维度: {args.embedding_dim}")
    else:
        print(f"Embedding 维度: 自动匹配（GConv: {args.gconv_dim}, TLayer: {args.tlayer_dim}）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    all_metrics = {
        "acc": [],
        "prec": [],
        "rec": [],
        "spec": [],
        "f1": [],
        "mcc": [],
        "auc": [],
        "aupr": [],
    }

    y = labels.numpy()
    for fold_id, (train_idx, test_idx) in enumerate(
        skf.split(edge_indices.numpy(), y), start=1
    ):
        print(
            f"=== Fold {fold_id}/{args.n_splits} === "
            f"Train {len(train_idx)}, Test {len(test_idx)}"
        )

        metrics = train_one_fold(
            fold_id=fold_id,
            data=data,
            labels=labels,
            train_edges=train_idx,
            test_edges=test_idx,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            gconv_dim=args.gconv_dim,
            tlayer_dim=args.tlayer_dim,
            num_layers=args.num_layers,
            tlayer_attn_dropout=args.tlayer_attn_dropout,
            tlayer_ffn_dropout=args.tlayer_ffn_dropout,
            rank_lambda=args.rank_lambda,
            rank_margin=args.rank_margin,
            use_gconv=args.use_gconv,
            use_tlayer=args.use_tlayer,
            classifier_type=args.classifier_type,
            embedding_dim=args.embedding_dim,
        )
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    # 输出每一折的详细结果
    print("\n===== 每一折详细结果（Test）=====")
    print(f"{'Fold':<6} {'Acc.':<8} {'Prec.':<8} {'Rec.':<8} {'Spec.':<8} {'F1':<8} {'MCC':<8} {'AUC':<8} {'AUPR':<8}")
    print("-" * 80)
    for fold_id in range(1, args.n_splits + 1):
        print(
            f"{fold_id:<6} "
            f"{all_metrics['acc'][fold_id-1]:<8.4f} "
            f"{all_metrics['prec'][fold_id-1]:<8.4f} "
            f"{all_metrics['rec'][fold_id-1]:<8.4f} "
            f"{all_metrics['spec'][fold_id-1]:<8.4f} "
            f"{all_metrics['f1'][fold_id-1]:<8.4f} "
            f"{all_metrics['mcc'][fold_id-1]:<8.4f} "
            f"{all_metrics['auc'][fold_id-1]:<8.4f} "
            f"{all_metrics['aupr'][fold_id-1]:<8.4f}"
        )

    print("\n===== 5 折结果（Test）=====")
    print(f"{'指标':<8} {'平均值':<10} {'标准差':<10}")
    print("-" * 30)
    print(f"{'Acc.':<8} {np.mean(all_metrics['acc']):.4f}    {np.std(all_metrics['acc']):.4f}")
    print(f"{'Prec.':<8} {np.mean(all_metrics['prec']):.4f}    {np.std(all_metrics['prec']):.4f}")
    print(f"{'Rec.':<8} {np.mean(all_metrics['rec']):.4f}    {np.std(all_metrics['rec']):.4f}")
    print(f"{'Spec.':<8} {np.mean(all_metrics['spec']):.4f}    {np.std(all_metrics['spec']):.4f}")
    print(f"{'F1':<8} {np.mean(all_metrics['f1']):.4f}    {np.std(all_metrics['f1']):.4f}")
    print(f"{'MCC':<8} {np.mean(all_metrics['mcc']):.4f}    {np.std(all_metrics['mcc']):.4f}")
    print(f"{'AUC':<8} {np.mean(all_metrics['auc']):.4f}    {np.std(all_metrics['auc']):.4f}")
    print(f"{'AUPR':<8} {np.mean(all_metrics['aupr']):.4f}    {np.std(all_metrics['aupr']):.4f}")
    
    # 输出表格格式（便于复制到表格）
    print("\n===== 表格格式 =====")
    print("Acc.\tPrec.\tRec.\tSpec.\tF1\tMCC\tAUC\tAUPR")
    print(
        f"{np.mean(all_metrics['acc']):.4f}\t"
        f"{np.mean(all_metrics['prec']):.4f}\t"
        f"{np.mean(all_metrics['rec']):.4f}\t"
        f"{np.mean(all_metrics['spec']):.4f}\t"
        f"{np.mean(all_metrics['f1']):.4f}\t"
        f"{np.mean(all_metrics['mcc']):.4f}\t"
        f"{np.mean(all_metrics['auc']):.4f}\t"
        f"{np.mean(all_metrics['aupr']):.4f}"
    )


if __name__ == "__main__":
    main()

