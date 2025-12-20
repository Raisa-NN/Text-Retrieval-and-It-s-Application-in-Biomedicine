# -*- coding: utf-8 -*-

import json
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b"


def truncate(text, head=600, tail=200):
    """
    Keep beginning and end of long patient narratives to reduce prompt size while
    still preserving late details (sometimes diagnoses/outcomes appear at the end).
    """
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= head + tail:
        return t
    return t[:head] + " ... " + t[-tail:]


def build_rerank_prompt(query_uid, query_text, candidates_by_uid):
    """
    Prompt: ask model to rank candidate UIDs by similarity to the query patient.
    Force JSON-only output for easy parsing.
    """
    cand_lines = []
    for uid, txt in candidates_by_uid.items():
        cand_lines.append(f"- uid: {uid}\n  text: {truncate(txt)}")

    return (
        "You are ranking similar patient cases.\n"
        "Rank the candidates from most similar to least similar.\n"
        "Return ONLY valid JSON in this schema:\n"
        '{"ranked_uids": ["uid1","uid2","uid3","uid4","uid5"]}\n\n'
        f"Query uid: {query_uid}\n"
        f"Query text: {truncate(query_text)}\n\n"
        "Candidates:\n" + "\n".join(cand_lines)
    )


def ollama_generate(prompt, temperature=0.0, timeout=300):
    """
    Call Ollama local API and return response text.
    num_predict caps response length to reduce timeouts.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 200
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def rerank_one(query_uid, uid_to_text, candidate_uids, top_k=5):
    """
    Rerank candidate_uids for one query patient using Ollama.
    If the LLM fails, fall back to BM25 ordering.
    """
    candidates_by_uid = {
        uid: uid_to_text[uid]
        for uid in candidate_uids
        if uid in uid_to_text and uid != query_uid
    }
    if not candidates_by_uid:
        return []

    prompt = build_rerank_prompt(query_uid, uid_to_text.get(query_uid, ""), candidates_by_uid)

    try:
        out_text = ollama_generate(prompt, temperature=0.0, timeout=300)
    except Exception:
        # fallback: just return the original candidate ordering (BM25 order)
        return [u for u in candidate_uids if u != query_uid][:top_k]

    try:
        parsed = json.loads(out_text)
        ranked = parsed.get("ranked_uids", [])
        if not isinstance(ranked, list):
            ranked = []
    except Exception:
        ranked = []

    cand_set = set(candidates_by_uid.keys())
    ranked = [u for u in ranked if u in cand_set]

    # Pad if model returns fewer than top_k
    if len(ranked) < top_k:
        for u in candidate_uids:
            if u != query_uid and u not in ranked:
                ranked.append(u)

    return ranked[:top_k]


def rerank_all(uids, raw_texts, candidates_dict, top_k=5):
    """
    Rerank for all query patients in uids.
    candidates_dict: uid -> list of candidate uids (e.g., BM25 top-20)
    """
    uid_to_text = dict(zip(uids, raw_texts))
    out = {}

    for i, uid in enumerate(uids, start=1):
        out[uid] = rerank_one(uid, uid_to_text, candidates_dict.get(uid, []), top_k=top_k)

        if i % 5 == 0 or i == len(uids):
            print(f"LLM reranking: processed {i}/{len(uids)}")

    return out

