# -*- coding: utf-8 -*-
"""
Main execution script for Programming Assignment 2
This script orchestrates the full retrieval and evaluation pipeline:

  Task 1:
    - Load patient JSON files
    - Preprocess patient text
    - Retrieve top-5 similar patients using:
        * BM25 lexical retrieval
        * LLM-based retrieval (via reranking of BM25 subset)

  Task 2:
    - Evaluate retrieval results against gold-standard labels
    - Compute precision@5 and recall@5 for each patient
    - Save per-patient metrics and dataset-level averages

The script is designed to be run from the command line, for example:

    python main.py ./PMC_Patients --sample 100

"""

import os
import argparse
import preprocess_utils
from io_utils import read_patient_jsons, save_json
from bm25_retriever import bm25_retrieve, eval_retrieval

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--no_llm", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Read JSONs
    uids, raw_texts, gold = read_patient_jsons(args.input_dir, limit=args.sample)

    # 2) Preprocess
    processed_docs = [' '.join(preprocess_utils.pre_process(text)) for text in raw_texts]

    # 3) BM25 retrieval
    bm25_top5 = bm25_retrieve(processed_docs, uids, top_k=5)

    # 5) Task1 output
    task1 = {uid: {
        "similar_patients_bm25": bm25_top5.get(uid, [])#,
       # "similar_patients_llm": llm_top5.get(uid, []) if llm_top5 else []
    } for uid in uids}
    save_json(task1, os.path.join(args.out_dir, "task1_retrieval.json"))

    # 6) Task2 evaluation
    bm25_per, bm25_avg = eval_retrieval(bm25_top5, gold, k=5)
    out_eval = {"bm25": {"per_patient": bm25_per, "summary": bm25_avg}}

   
    save_json(out_eval, os.path.join(args.out_dir, "task2_eval.json"))

if __name__ == "__main__":
    main()
'''
    # 4) LLM retrieval (recommended approach: rerank BM25 candidates)
    llm_top5 = {}
    if not args.no_llm:
        llm_top5 = llm_rerank_over_bm25_candidates(uids, raw_texts, bm25_top5)  # you’ll implement this

 if llm_top5:
     llm_per, llm_avg = eval_retrieval(llm_top5, gold, k=5)
     out_eval["llm"] = {"per_patient": llm_per, "summary": llm_avg}

'''