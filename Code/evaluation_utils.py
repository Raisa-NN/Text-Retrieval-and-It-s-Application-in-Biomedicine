# -*- coding: utf-8 -*-
"""
Responsibilities:
  - Precision@k and Recall@k evaluation
  - Aggregated evaluation
"""
import numpy as np

def precision_recall_at_k(retrieved, gold_list, k=5):
    """
    Compute precision@k and recall@k for a single query.
    Args:
        retrieved: Ranked list of retrieved IDs.
        gold_list: List of relevant gold standard IDs.
        k: Rank cutoff.
    Relevance is treated as binary: any patient UID present in the gold
    list is considered relevant, regardless of its original relevance score.
    """
    
    topk = retrieved[:k]
    # Convert gold list to set for fast membership tests
    gold = set(gold_list)
    # Count how many of the top-k retrieved items appear in the gold list
    hits = sum(1 for x in topk if x in gold)
    # Of the k retrieved items, what fraction are relevant?
    p = hits / float(k) if k else 0.0
    # Of all relevant (gold) items, what fraction were retrieved in the top-k?
    r = hits / float(len(gold)) if gold else 0.0
    return p, r

def eval_retrieval(retrieval_results, gold, k=5):
    """
    This function computes precision@k and recall@k for each patient,
    then aggregates the results by reporting average precision@k and
    average recall@k across all patients.
    Notes:
        - Patients with no gold labels receive recall@k = 0 by convention.
        - All patients contribute equally to the average metrics.
    """
    
    per_patient = {}
    ps, rs = [], []
    for uid, retrieved in retrieval_results.items():
        p, r = precision_recall_at_k(retrieved, gold.get(uid, []), k=k)
        per_patient[uid] = {"precision@5": round(p, 4), "recall@5": round(r, 4)}
        ps.append(p); rs.append(r)

    summary = {"avg_precision@5": round(float(np.mean(ps)) if ps else 0.0, 4),
               "avg_recall@5": round(float(np.mean(rs)) if rs else 0.0, 4)}
    return per_patient, summary
