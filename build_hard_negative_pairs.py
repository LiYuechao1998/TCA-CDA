import os
import argparse

import numpy as np
import pandas as pd


def build_hard_negatives(
    data_dir: str,
    k_neg_per_pos: int = 1,
    strategy: str = "disease_similarity",
    difficulty: float = 1.0,
    min_similarity: float = None,
    max_similarity: float = None,
    ensure_diversity: bool = False,
) -> None:

    circ_128 = os.path.join(data_dir, "circRNA_bert_128.csv")
    circ_32 = os.path.join(data_dir, "circRNA_bert_32.csv")
    dis_128 = os.path.join(data_dir, "disease_GIPK_128.csv")
    dis_32 = os.path.join(data_dir, "disease_GIPK_32.csv")

    circ_path = circ_128 if os.path.exists(circ_128) else circ_32
    dis_path = dis_128 if os.path.exists(dis_128) else dis_32

    pair_path = os.path.join(data_dir, "sample_pairs.csv")
    interaction_path = os.path.join(data_dir, "interaction.csv")
    use_interaction_only = False
    
    if not os.path.exists(pair_path):
        pair_path = interaction_path
        use_interaction_only = True
    else:
        if os.path.exists(interaction_path):
            try:
                sample_pairs_df = pd.read_csv(pair_path, header=None)
                interaction_df = pd.read_csv(interaction_path, header=None)
                if len(sample_pairs_df) == len(interaction_df):
                    if sample_pairs_df.equals(interaction_df):
                        use_interaction_only = True
            except:
                pass
    
    feat_path = os.path.join(data_dir, "export_features", "node_features.csv")

    if not os.path.exists(pair_path):
        raise FileNotFoundError(f"")
    
    circ_df = pd.read_csv(circ_path, header=None)
    dis_df = pd.read_csv(dis_path, header=None)
    pair_df = pd.read_csv(pair_path, header=None, names=["circ", "disease"])

    circ_ids = circ_df.iloc[:, 0].astype(str).tolist()
    dis_ids = dis_df.iloc[:, 0].astype(str).tolist()

    circ_id2idx = {cid: i for i, cid in enumerate(circ_ids)}
    num_circ = len(circ_ids)
    dis_id2idx = {did: i for i, did in enumerate(dis_ids)}

    if os.path.exists(feat_path):
        X = np.loadtxt(feat_path, delimiter=",")
        if X.shape[0] < num_circ + len(dis_ids):
            print(
                f""
            )
            circ_feat = circ_df.iloc[:, 1:].to_numpy(dtype=np.float32)
            dis_feat = dis_df.iloc[:, 1:].to_numpy(dtype=np.float32)
            feat_circ = circ_feat
            feat_dis = dis_feat
        else:
            feat_circ = X[:num_circ]             # (num_circ, d)
            feat_dis = X[num_circ : num_circ + len(dis_ids)]  # (num_dis, d)
    else:
        circ_feat = circ_df.iloc[:, 1:].to_numpy(dtype=np.float32)
        dis_feat = dis_df.iloc[:, 1:].to_numpy(dtype=np.float32)
        feat_circ = circ_feat
        feat_dis = dis_feat

    if use_interaction_only:
        pos_pairs = pair_df.copy()
    else:
        pos_len = len(pair_df) // 2
        pos_pairs = pair_df.iloc[:pos_len].copy()

    circ_pos_dis = {i: set() for i in range(num_circ)}
    for _, row in pos_pairs.iterrows():
        c = str(row["circ"])
        d = str(row["disease"])
        if c not in circ_id2idx or d not in dis_id2idx:
            continue
        u = circ_id2idx[c]
        v = dis_id2idx[d]
        circ_pos_dis[u].add(v)

    hard_neg_rows = []
    
    if strategy == "disease_similarity":
        feat_dis_norm = feat_dis / (np.linalg.norm(feat_dis, axis=1, keepdims=True) + 1e-8)
        sim_mat = feat_dis_norm @ feat_dis_norm.T  # (num_dis, num_dis)

        for _, row in pos_pairs.iterrows():
            c = str(row["circ"])
            d = str(row["disease"])
            if c not in circ_id2idx or d not in dis_id2idx:
                continue
            u = circ_id2idx[c]
            v = dis_id2idx[d]

            sims = sim_mat[v]
            cand_indices = np.argsort(-sims)

            valid_candidates = []
            for dis_idx in cand_indices:
                if dis_idx == v:
                    continue
                if dis_idx in circ_pos_dis[u]:
                    continue
                valid_candidates.append(dis_idx)
            
            if len(valid_candidates) == 0:
                continue

            num_candidates = len(valid_candidates)
            if difficulty >= 1.0:
                candidate_pool = valid_candidates[:min(k_neg_per_pos * 3, num_candidates)]
            elif difficulty <= 0.0:
                candidate_pool = valid_candidates
            else:
                top_k = max(1, int(k_neg_per_pos * difficulty))
                if difficulty > 0.95:
                    range_size = max(1, int((1 - difficulty) * num_candidates * 0.02))
                elif difficulty > 0.9:
                    range_size = max(1, int((1 - difficulty) * num_candidates * 0.05))
                elif difficulty > 0.7:
                    range_size = max(1, int((1 - difficulty) * num_candidates * 0.1))
                else:
                    range_size = max(1, int((1 - difficulty) * num_candidates * 0.5))
                end_idx = min(top_k + range_size, num_candidates)
                if end_idx > top_k:
                    candidate_pool = valid_candidates[top_k:min(end_idx, num_candidates)]
                else:
                    candidate_pool = valid_candidates[:min(k_neg_per_pos * 3, num_candidates)]

            if ensure_diversity and len(candidate_pool) > k_neg_per_pos:
                selected_range = []
                remaining_candidates = candidate_pool.copy()

                if len(remaining_candidates) > 0:
                    selected_range.append(remaining_candidates[0])
                    remaining_candidates.remove(remaining_candidates[0])

                while len(selected_range) < k_neg_per_pos and len(remaining_candidates) > 0:
                    if len(selected_range) == 0:
                        selected_range.append(remaining_candidates[0])
                        remaining_candidates.remove(remaining_candidates[0])
                        continue

                    best_candidate = None
                    min_avg_sim = float('inf')
                    
                    for cand_idx in remaining_candidates:
                        avg_sim = np.mean([sim_mat[cand_idx][sel_idx] for sel_idx in selected_range])
                        if avg_sim < min_avg_sim:
                            min_avg_sim = avg_sim
                            best_candidate = cand_idx
                    
                    if best_candidate is not None:
                        selected_range.append(best_candidate)
                        remaining_candidates.remove(best_candidate)
                    else:
                        break
            else:
                if len(candidate_pool) <= k_neg_per_pos:
                    selected_range = candidate_pool
                else:
                    if difficulty <= 0.0:
                        selected_indices = np.random.choice(len(candidate_pool), size=k_neg_per_pos, replace=False)
                        selected_range = [candidate_pool[i] for i in selected_indices]
                    else:
                        selected_range = candidate_pool[:k_neg_per_pos]
            
            for dis_idx in selected_range:
                hard_neg_rows.append((c, dis_ids[dis_idx]))
                    
    elif strategy == "pair_similarity":
        feat_circ_norm = feat_circ / (np.linalg.norm(feat_circ, axis=1, keepdims=True) + 1e-8)
        feat_dis_norm = feat_dis / (np.linalg.norm(feat_dis, axis=1, keepdims=True) + 1e-8)
        sim_mat = feat_dis_norm @ feat_dis_norm.T  # (num_dis, num_dis)
        
        for _, row in pos_pairs.iterrows():
            c = str(row["circ"])
            d = str(row["disease"])
            if c not in circ_id2idx or d not in dis_id2idx:
                continue
            u = circ_id2idx[c]
            v = dis_id2idx[d]
            

            pos_emb = np.concatenate([feat_circ_norm[u], feat_dis_norm[v]])
            

            pair_sims = []
            for dis_idx in range(len(dis_ids)):
                if dis_idx == v:
                    continue
                if dis_idx in circ_pos_dis[u]:
                    continue  #
                neg_emb = np.concatenate([feat_circ_norm[u], feat_dis_norm[dis_idx]])
                sim = np.dot(pos_emb, neg_emb)
                if min_similarity is not None and sim < min_similarity:
                    continue
                if max_similarity is not None and sim > max_similarity:
                    continue
                pair_sims.append((sim, dis_idx))

            pair_sims.sort(reverse=True)
            
            if len(pair_sims) == 0:
                continue

            num_candidates = len(pair_sims)
            if difficulty >= 1.0:
                candidate_pool = pair_sims[:min(k_neg_per_pos * 3, num_candidates)]
            elif difficulty <= 0.0:
                candidate_pool = pair_sims
            else:
                top_k = max(1, int(k_neg_per_pos * difficulty))
                if difficulty > 0.95:
                    range_size = max(1, int((1 - difficulty) * num_candidates * 0.02))
                elif difficulty > 0.9:
                    range_size = max(1, int((1 - difficulty) * num_candidates * 0.05))
                elif difficulty > 0.75:
                    range_size = max(1, int((1 - difficulty) * num_candidates * 0.1))
                elif difficulty > 0.65:
                    range_size = max(1, int((1 - difficulty) * num_candidates * 0.2))
                else:
                    range_size = max(1, int((1 - difficulty) * num_candidates * 0.4))
                end_idx = min(top_k + range_size, num_candidates)
                if end_idx > top_k:
                    candidate_pool = pair_sims[top_k:min(end_idx, num_candidates)]
                else:
                    candidate_pool = pair_sims[:min(k_neg_per_pos * 3, num_candidates)]

            if ensure_diversity and len(candidate_pool) > k_neg_per_pos:
                selected_pairs = []
                remaining_pairs = candidate_pool.copy()

                if len(remaining_pairs) > 0:
                    selected_pairs.append(remaining_pairs[0])
                    remaining_pairs.remove(remaining_pairs[0])

                while len(selected_pairs) < k_neg_per_pos and len(remaining_pairs) > 0:
                    if len(selected_pairs) == 0:
                        selected_pairs.append(remaining_pairs[0])
                        remaining_pairs.remove(remaining_pairs[0])
                        continue

                    best_pair = None
                    min_avg_sim = float('inf')
                    
                    for sim_score, cand_dis_idx in remaining_pairs:
                        selected_dis_indices = [dis_idx for _, dis_idx in selected_pairs]
                        avg_sim = np.mean([sim_mat[cand_dis_idx][sel_dis_idx] for sel_dis_idx in selected_dis_indices])
                        if avg_sim < min_avg_sim:
                            min_avg_sim = avg_sim
                            best_pair = (sim_score, cand_dis_idx)
                    
                    if best_pair is not None:
                        selected_pairs.append(best_pair)
                        remaining_pairs.remove(best_pair)
                    else:
                        break
            else:
                if len(candidate_pool) <= k_neg_per_pos:
                    selected_pairs = candidate_pool
                else:
                    if difficulty <= 0.0:
                        selected_indices = np.random.choice(len(candidate_pool), size=k_neg_per_pos, replace=False)
                        selected_pairs = [candidate_pool[i] for i in selected_indices]
                    else:
                        selected_pairs = candidate_pool[:k_neg_per_pos]
            
            for sim, dis_idx in selected_pairs:
                hard_neg_rows.append((c, dis_ids[dis_idx]))
                    
    elif strategy == "mixed":
        feat_dis_norm = feat_dis / (np.linalg.norm(feat_dis, axis=1, keepdims=True) + 1e-8)
        feat_circ_norm = feat_circ / (np.linalg.norm(feat_circ, axis=1, keepdims=True) + 1e-8)
        sim_mat = feat_dis_norm @ feat_dis_norm.T
        
        k_disease = max(1, k_neg_per_pos // 2)
        k_pair = k_neg_per_pos - k_disease
        
        for _, row in pos_pairs.iterrows():
            c = str(row["circ"])
            d = str(row["disease"])
            if c not in circ_id2idx or d not in dis_id2idx:
                continue
            u = circ_id2idx[c]
            v = dis_id2idx[d]
            
            added = 0
            used_dis = set()

            if k_disease > 0:
                sims = sim_mat[v]
                cand_indices = np.argsort(-sims)

                valid_candidates = []
                for dis_idx in cand_indices:
                    if dis_idx == v or dis_idx in circ_pos_dis[u] or dis_idx in used_dis:
                        continue
                    valid_candidates.append(dis_idx)
                
                    if len(valid_candidates) > 0:
                        num_candidates = len(valid_candidates)
                        if difficulty >= 1.0:
                            selected_range = valid_candidates[:k_disease]
                        elif difficulty <= 0.0:
                            selected_indices = np.random.choice(num_candidates, size=min(k_disease, num_candidates), replace=False)
                            selected_range = [valid_candidates[i] for i in selected_indices]
                        else:
                            top_k = max(1, int(k_disease * difficulty))
                            if difficulty > 0.9:

                                range_size = max(1, int((1 - difficulty) * num_candidates * 0.05))
                            elif difficulty > 0.7:

                                range_size = max(1, int((1 - difficulty) * num_candidates * 0.1))
                            else:
                                range_size = max(1, int((1 - difficulty) * num_candidates * 0.5))
                            end_idx = min(top_k + range_size, num_candidates)
                        if end_idx > top_k:
                            selected_indices = np.random.choice(
                                range(top_k, end_idx), 
                                size=min(k_disease, end_idx - top_k), 
                                replace=False
                            )
                            selected_range = [valid_candidates[i] for i in selected_indices]
                        else:
                            selected_range = valid_candidates[:k_disease]
                    
                    for dis_idx in selected_range:
                        hard_neg_rows.append((c, dis_ids[dis_idx]))
                        used_dis.add(dis_idx)
                        added += 1
                        if added >= k_disease:
                            break
            

            if k_pair > 0 and added < k_neg_per_pos:
                pos_emb = np.concatenate([feat_circ_norm[u], feat_dis_norm[v]])
                pair_sims = []
                for dis_idx in range(len(dis_ids)):
                    if dis_idx == v or dis_idx in circ_pos_dis[u] or dis_idx in used_dis:
                        continue
                    neg_emb = np.concatenate([feat_circ_norm[u], feat_dis_norm[dis_idx]])
                    sim = np.dot(pos_emb, neg_emb)
                    pair_sims.append((sim, dis_idx))
                
                if len(pair_sims) > 0:
                    pair_sims.sort(reverse=True)
                    

                    num_candidates = len(pair_sims)
                    if difficulty >= 1.0:
                        selected_pairs = pair_sims[:k_pair]
                    elif difficulty <= 0.0:
                        selected_indices = np.random.choice(num_candidates, size=min(k_pair, num_candidates), replace=False)
                        selected_pairs = [pair_sims[i] for i in selected_indices]
                    else:
                        top_k = max(1, int(k_pair * difficulty))

                        if difficulty > 0.95:

                            range_size = max(1, int((1 - difficulty) * num_candidates * 0.02))
                        elif difficulty > 0.9:

                            range_size = max(1, int((1 - difficulty) * num_candidates * 0.05))
                        elif difficulty > 0.7:

                            range_size = max(1, int((1 - difficulty) * num_candidates * 0.1))
                        else:
                            range_size = max(1, int((1 - difficulty) * num_candidates * 0.5))
                        end_idx = min(top_k + range_size, num_candidates)
                        if end_idx > top_k:
                            selected_indices = np.random.choice(
                                range(top_k, end_idx), 
                                size=min(k_pair, end_idx - top_k), 
                                replace=False
                            )
                            selected_pairs = [pair_sims[i] for i in selected_indices]
                        else:
                            selected_pairs = pair_sims[:k_pair]
                    
                    for sim, dis_idx in selected_pairs:
                        hard_neg_rows.append((c, dis_ids[dis_idx]))
                        added += 1
                        if added >= k_neg_per_pos:
                            break
    else:
        raise ValueError(f": {strategy}，: disease_similarity, pair_similarity, mixed")

    hardneg_df = pd.DataFrame(hard_neg_rows, columns=["circ", "disease"])


    new_pairs = pd.concat([pos_pairs, hardneg_df], axis=0, ignore_index=True)
    out_path = os.path.join(data_dir, "sample_pairs_hardneg.csv")
    new_pairs.to_csv(out_path, header=False, index=False)



def main():
    parser = argparse.ArgumentParser(
        description="sample_pairs_hardneg.csv"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help=" circRNA_bert_*.csv / disease_GIPK_*.csv / sample_pairs.csv ",
    )
    parser.add_argument("--k_neg_per_pos", type=int, default=1,
                       help="")
    parser.add_argument("--strategy", type=str, default="disease_similarity",
                       choices=["disease_similarity", "pair_similarity", "mixed"],
                       help="")
    parser.add_argument("--difficulty", type=float, default=1.0,
                       help="")
    parser.add_argument("--min_similarity", type=float, default=None,
                       help="")
    parser.add_argument("--max_similarity", type=float, default=None,
                       help="")
    parser.add_argument("--ensure_diversity", action="store_true",
                       help="")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if os.path.isabs(args.data_dir):
        data_dir = args.data_dir
    else:

        data_dir = os.path.join(script_dir, args.data_dir)
        if not os.path.exists(data_dir):

            parent_dir = os.path.dirname(script_dir)
            data_dir = os.path.join(parent_dir, args.data_dir)

    build_hard_negatives(
        data_dir, 
        k_neg_per_pos=args.k_neg_per_pos, 
        strategy=args.strategy, 
        difficulty=args.difficulty,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity,
        ensure_diversity=args.ensure_diversity
    )


if __name__ == "__main__":
    main()

