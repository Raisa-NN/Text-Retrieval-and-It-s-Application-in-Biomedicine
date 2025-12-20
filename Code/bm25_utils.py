# -*- coding: utf-8 -*-
"""
Responsibilities:
  - BM25-based lexical retrieval: probabilistic ranking function with TF saturation + length normalization built in.
"""

from rank_bm25 import BM25Okapi

def bm25_retrieve(processed_docs, uids, k1=1.5, b=0.75, top_k=10):
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

