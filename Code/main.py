# -*- coding: utf-8 -*-

import os
import argparse
import preprocess_utils
from io_utils import read_patient_jsons, save_json
from bm25_utils import bm25_retrieve
from word2vec_utils import word2vec_retrieve
from evaluation_utils import eval_retrieval
from llm_utils import rerank_all  # Ollama-based reranker


def main():
    parser = argparse.ArgumentParser(description="HI 744: BM25 + Word2Vec + Ollama reranking")
    parser.add_argument("input_dir", type=str, help="Folder containing patient JSON files")
    args = parser.parse_args()

   
    # I had to do this because it took forever to run the code for the entire corpus
    # To run full dataset, please change 'SAMPLE_SIZE' values to 167034 
    SAMPLE_SIZE = 10
    RANDOM_SEED = 42
    BM25_CAND_POOL = 10
    TOP_K = 5

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # 1) Read a reproducible random sample of 100 patients (+ gold)
    uids, raw_texts, gold = read_patient_jsons(
        args.input_dir,
        limit=SAMPLE_SIZE,
        seed=RANDOM_SEED
    )
    print("Loaded:", len(uids))
    print("First 5 UIDs:", uids[:5])

    # Useful lookup (uid -> raw text)
    #uid_to_text = dict(zip(uids, raw_texts))

    # 2) Preprocess for lexical/embedding methods
    processed_docs = [" ".join(preprocess_utils.pre_process(t)) for t in raw_texts]

    # 3) BM25 candidate pool (top-20 for every UID)
    bm25_top20 = bm25_retrieve(processed_docs, uids, top_k=BM25_CAND_POOL)

    # BM25 final output (top-5 for every UID)
    bm25_top5 = {u: bm25_top20.get(u, [])[:TOP_K] for u in uids}

    # 4) Word2Vec output (top-5 for every UID)
    w2v_top5 = word2vec_retrieve(processed_docs, uids, top_k=TOP_K)

    # 5) LLM rerank BM25 top-20 -> top-5 for every UID (all 100 queries)
    # llm_top5 = rerank_all(
    #     uids=uids,
    #     raw_texts=raw_texts,
    #     candidates_dict=bm25_top20,
    #     top_k=TOP_K
    # )

    # 6) Task1 output: top-5 for EVERY UID for all three methods
    task1 = {}
    for uid in uids:
        task1[uid] = {
            "similar_patients_bm25": bm25_top5.get(uid, []),
            "similar_patients_word2vec": w2v_top5.get(uid, []),
            #"similar_patients_llm": llm_top5.get(uid, [])
        }
    save_json(task1, os.path.join(out_dir, "task1_retrieval.json"))

    # 7) Filter gold labels to only those UIDs present in the sample
    # This avoids “forced zeros” when gold patients exist outside the sampled corpus.
    uid_set = set(uids)
    gold_in_sample = {u: [g for g in gold.get(u, []) if g in uid_set] for u in uids}

    # 8) Task2 evaluation 
    bm25_per, bm25_avg = eval_retrieval(bm25_top5, gold_in_sample, k=TOP_K)
    w2v_per, w2v_avg = eval_retrieval(w2v_top5, gold_in_sample, k=TOP_K)
   # llm_per, llm_avg = eval_retrieval(llm_top5, gold_in_sample, k=TOP_K)

    out_eval = {
    "bm25": {"per_patient": bm25_per, "summary": bm25_avg},
    "word2vec": {"per_patient": w2v_per, "summary": w2v_avg},
   # "llm": {"per_patient": llm_per, "summary": llm_avg},
    }
    save_json(out_eval, os.path.join(out_dir, "task2_eval.json"))
    print("DONE")

if __name__ == "__main__":
    main()


