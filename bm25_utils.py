# -*- coding: utf-8 -*-
"""
Responsibilities:
  - BM25-based lexical retrieval: probabilistic ranking function with TF saturation + length normalization built in.
  - Precision@k and Recall@k evaluation
  - Aggregated evaluation
"""
import numpy as np
from rank_bm25 import BM25Okapi

def bm25_retrieve(processed_docs, uids, k1=1.5, b=0.75, top_k=15):
    """
    Retrieve top-k similar patients for each patient using BM25.
    This treats each patient document as a query against the entire corpus
    and returns ranked neighbors (excluding the patient itself).

    Args:
        processed_docs (list[str]): List of whitespace-tokenized documents (strings).
        uids (list[str]): Document IDs aligned with processed_docs.
        k1 (float): BM25 term-frequency saturation parameter [1.2, 2.0].
        b (float): BM25 length normalization parameter from [0, 1].
        b=0 disables length normalization, b=1 applies full normalization.
        top_k (int): Number of similar patients to retrieve per query.

    Returns:
        dict[str, list[str]]:
            Dictionary mapping each patient UID to a list of the
            top-k retrieved patient UIDs, ordered by descending BM25 score.
    """

    tokenized = [d.split() for d in processed_docs]
    bm25 = BM25Okapi(tokenized, k1=k1, b=b)

    results = {}  # uid -> [top similar uids]
    for i, uid in enumerate(uids):
        scores = bm25.get_scores(processed_docs[i].split())
        pairs = [(uids[j], float(scores[j])) for j in range(len(uids)) if uids[j] != uid]
        pairs.sort(key=lambda x: x[1], reverse=True)
        results[uid] = [u for u, _ in pairs[:top_k]]
    return results

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
        per_patient[uid] = {"precision_5": round(p, 4), "recall_5": round(r, 4)}
        ps.append(p); rs.append(r)

    summary = {"avg_precision_5": round(float(np.mean(ps)) if ps else 0.0, 4),
               "avg_recall_5": round(float(np.mean(rs)) if rs else 0.0, 4)}
    return per_patient, summary
