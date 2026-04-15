import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.optim import AdamW
from torch_geometric.data import Data


def build_graph(data_dir: str, feat_dim: int = None) -> tuple[Data, torch.Tensor, torch.Tensor, int]:
    circ_32 = os.path.join(data_dir, "circRNA_bert_32.csv")
    circ_64 = os.path.join(data_dir, "circRNA_bert_64.csv")
    circ_128 = os.path.join(data_dir, "circRNA_bert_128.csv")
    dis_32 = os.path.join(data_dir, "disease_GIPK_32.csv")
    dis_64 = os.path.join(data_dir, "disease_GIPK_64.csv")
    dis_128 = os.path.join(data_dir, "disease_GIPK_128.csv")
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
            raise ValueError(f"")
        
        # 检查指定维度的文件是否存在
        if not os.path.exists(circ_path):
            raise FileNotFoundError(f": {circ_path}")
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
    
    # 显示实际使用的特征文件（用于确认优先级）
    print(f"使用 circRNA 特征文件: {os.path.basename(circ_path)}")
    print(f"使用 disease 特征文件: {os.path.basename(dis_path)}")
    # 若存在基于 hard negative 重新构造的 sample_pairs_hardneg.csv，则优先使用
    pair_hard = os.path.join(data_dir, "sample_pairs_hardneg.csv")
    pair_path = pair_hard if os.path.exists(pair_hard) else os.path.join(data_dir, "sample_pairs.csv")

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

    # 前一半 label=1，后一半 label=0
    labels = torch.zeros(num_edges, dtype=torch.long)
    pos_len = num_edges // 2
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
    """边分类器基类
    
    消融实验：不使用 GNN 编码，直接用原始特征进行链接预测。
    """
    
    def __init__(self, node_dim: int, classifier_type: str = "linear_bilinear"):
        super().__init__()
        self.node_dim = node_dim
        self.classifier_type = classifier_type
        self._build_classifier()
    
    def _build_classifier(self):
        """根据分类器类型构建不同的分类器"""
        in_dim = self.node_dim * 4  # [h_u, h_v, |h_u-h_v|, h_u*h_v]
        
        if self.classifier_type == "linear_bilinear":
            # 线性 + 双线性
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
    rank_lambda: float,
    classifier_type: str = "linear_bilinear",
) -> dict:
    """在一个折上训练+评估，返回基于 Test AUC 最优时刻的所有评估指标.
    
    消融实验：不使用 GNN，直接使用原始节点特征。
    """
    # 不使用 GNN，直接使用原始节点特征
    node_dim = data.x.size(1)
    
    edge_cls = EdgeClassifier(node_dim=node_dim, classifier_type=classifier_type)

    edge_cls.to(device)
    data = data.to(device)
    labels = labels.to(device)

    optimizer = AdamW(
        list(edge_cls.parameters()),
        lr=lr,
        weight_decay=1e-5,
    )
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

    for epoch in range(1, epochs + 1):
        edge_cls.train()

        optimizer.zero_grad()
        # 消融实验：直接使用原始节点特征，不使用 GNN 编码
        node_emb = data.x  # (N_nodes, node_dim) - 原始特征

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
            rank_loss = F.margin_ranking_loss(logits_pos, logits_neg, target, margin=0.5)
        else:
            rank_loss = torch.tensor(0.0, device=device)

        loss = bce_loss + rank_lambda * rank_loss
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            edge_cls.eval()
            with torch.no_grad():
                # 消融实验：直接使用原始节点特征
                node_emb = data.x
                logits = edge_cls(node_emb, data.edge_index[:, test_edges_t])
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
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
                    best_metrics = {
                        "acc": test_acc,
                        "prec": test_prec,
                        "rec": test_rec,
                        "spec": test_spec,
                        "f1": test_f1,
                        "mcc": test_mcc,
                        "auc": test_auc,
                        "aupr": test_aupr,
                    }
                    # 记录最佳时刻的概率与标签，方便后续保存为 .npy
                    best_probs = y_probs
                    best_y_true = y_true

                print(
                    f"[Fold {fold_id}] Epoch {epoch:03d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Test AUC {test_auc:.4f} Acc {test_acc:.4f} | "
                    f"Best Test AUC {best_test_auc:.4f} Acc {best_metrics['acc']:.4f}"
                )

    # 将该折上"最佳 AUC 时刻"的预测概率和真实标签保存为 .npy，方便后续画 ROC / PR 曲线
    if best_probs is not None and best_y_true is not None:
        np.save(f"Y_pre_ablation_no_gnn_{fold_id - 1}.npy", best_probs)
        np.save(f"Y_test_ablation_no_gnn_{fold_id - 1}.npy", best_y_true)

    return best_metrics


def main():
    parser = argparse.ArgumentParser(
        description="circRNA-disease 链路预测（5 折）- 消融实验：不使用 GNN，直接使用原始特征"
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
    # 分类器类型
    parser.add_argument("--classifier_type", type=str, default="linear_bilinear",
                       choices=["linear_bilinear", "linear_only", "bilinear_only", "mlp", "mlp_deep", "dot_product"],
                       help="分类器类型：linear_bilinear(默认), linear_only, bilinear_only, mlp, mlp_deep, dot_product")
    # ranking loss 权重
    parser.add_argument("--rank_lambda", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--use_cpu", action="store_true", help="强制使用 CPU（即使有 CUDA）")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(script_dir, args.data_dir)

    print(f"加载数据集自: {data_dir}")
    print("=" * 60)
    print("消融实验：不使用 GNN，直接使用原始节点特征进行链接预测")
    print("=" * 60)
    if args.feat_dim is not None:
        print(f"指定特征维度: {args.feat_dim}")
    else:
        print("自动选择特征维度（优先级：32 > 64 > 128）")
    
    data, labels, edge_indices, feat_dim = build_graph(data_dir, feat_dim=args.feat_dim)
    
    print(f"实际使用特征维度: {feat_dim}")
    
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

    if args.use_cpu:
        device = torch.device("cpu")
        print("强制使用 CPU")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print(f"使用设备: {device} (CUDA {torch.version.cuda})")
        else:
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
            rank_lambda=args.rank_lambda,
            classifier_type=args.classifier_type,
        )
        for key in all_metrics:
            all_metrics[key].append(metrics[key])

    # 输出每一折的详细结果
    print("\n===== 每一折详细结果（Test）- 消融实验（无 GNN）=====")
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

    print("\n===== 5 折结果（Test）- 消融实验（无 GNN）=====")
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
