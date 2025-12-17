# -*- coding: utf-8 -*-

import os
import argparse
import preprocess_utils
from io_utils import read_patient_jsons, save_json
from bm25_utils import bm25_retrieve, eval_retrieval
from llm_utils import rerank_all  # Ollama-based reranker


def main():
    parser = argparse.ArgumentParser(description="PMC Patients: BM25 + Ollama reranking")
    parser.add_argument("input_dir", type=str, help="Folder containing patient JSON files")
    parser.add_argument("--sample", type=int, default=None, help="Process only first N patients")
    parser.add_argument("--out_dir", type=str, default="outputs", help="Output folder")
    parser.add_argument("--no_llm", action="store_true", help="Disable LLM reranking")
    parser.add_argument("--mock_llm", action="store_true", help="Mock LLM (no Ollama calls) for testing")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Read data (+ gold labels)
    uids, raw_texts, gold = read_patient_jsons(args.input_dir, limit=args.sample)

    # 2) Preprocess (for BM25 only)
    processed_docs = [" ".join(preprocess_utils.pre_process(t)) for t in raw_texts]

    # 3) BM25 candidate pool: TOP-20
    bm25_top15 = bm25_retrieve(processed_docs, uids, top_k=15)

    # BM25 baseline output: TOP-5 (first 5 from the pool)
    bm25_top5 = {uid: bm25_top15.get(uid, [])[:5] for uid in uids}

    # 4) LLM rerank: rerank BM25 TOP-20 → TOP-5
    llm_top5 = {}
    if not args.no_llm:
        llm_top5 = rerank_all(
            uids=uids,
            raw_texts=raw_texts,
            candidates_dict=bm25_top15,  # <-- IMPORTANT: rerank the top-20 pool
            top_k=5,
            temperature=0.0,
            mock_llm=args.mock_llm,
        )

    # 5) Task 1 output
    task1 = {
        uid: {
            "similar_patients_bm25": bm25_top5.get(uid, []),
            "similar_patients_llm": llm_top5.get(uid, []) if llm_top5 else [],
            # Optional: keep pool for debugging / transparency
            "bm25_candidate_pool_top15": bm25_top15.get(uid, []),
        }
        for uid in uids
    }
    save_json(task1, os.path.join(args.out_dir, "task1_retrieval.json"))

    # 6) Task 2 evaluation (evaluate TOP-5 lists only)
    bm25_per, bm25_avg = eval_retrieval(bm25_top5, gold, k=5)
    out_eval = {"bm25": {"per_patient": bm25_per, "summary": bm25_avg}}

    if llm_top5:
        llm_per, llm_avg = eval_retrieval(llm_top5, gold, k=5)
        out_eval["llm"] = {"per_patient": llm_per, "summary": llm_avg}

    save_json(out_eval, os.path.join(args.out_dir, "task2_eval.json"))

    print("DONE")
    print("Wrote:", os.path.join(args.out_dir, "task1_retrieval.json"))
    print("Wrote:", os.path.join(args.out_dir, "task2_eval.json"))


if __name__ == "__main__":
    main()
