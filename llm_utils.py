# -*- coding: utf-8 -*-

import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b" # You can change to "llama3.1:8b" if you want higher quality (slower)  


def build_rerank_prompt(query_uid, query_text, candidates_by_uid):
    """
    Build a simple prompt that asks the local LLM to rank candidate patients.
    candidates_by_uid: dict {candidate_uid: candidate_text}
    """
    
    cand_lines = []
    for uid, text in candidates_by_uid.items():
        cand_lines.append(f"- uid: {uid}\n  text: {truncate_head_tail(text)}")

    prompt = (
        "You are ranking similar patient cases.\n"
        "Return ONLY a valid JSON object. Do not include any extra text.\n"
        "JSON schema:\n"
        '{"ranked_uids":["uid1", "..."], "rationale_short":{"uid1":"short reason", "...":"..."}}\n\n'
        f"Query uid: {query_uid}\n"
        f"Query text: {truncate_head_tail(query_text)}\n\n"
        "Candidates:\n"
        + "\n".join(cand_lines)
    )
    return prompt
    
    


def truncate_head_tail(text, head=200, tail=100):
    """Keep the beginning and end to reduce prompt length.
    Why head+tail?
   - Clinical narratives often end with diagnosis/outcome/treatment details.
   - Keeping only the beginning can drop the most important information."""
    t = (text or "").replace("\n", " ").strip()
    if len(t) <= head + tail:
        return t
    return t[:head] + " ... " + t[-tail:]


def ollama_generate(prompt, model=OLLAMA_MODEL, temperature=0.0):
    """
    Call Ollama local API and return text response.
    Requires: Ollama running locally.
    
    Notes:
    - timeout is higher because CPU inference can be slow.
    - temperature=0 encourages deterministic outputs.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
       r = requests.post(OLLAMA_URL, json=payload, timeout=600)
       r.raise_for_status()
       data = r.json()
       return (data.get("response", "") or "").strip()
    except requests.exceptions.Timeout:
       print("WARNING: Ollama timeout. Falling back to BM25 order for this query.")
       return ""
    except requests.exceptions.RequestException as e:
       print("WARNING: Ollama request failed:", e)
       return ""

def rerank_one(query_uid, uid_to_text, candidate_uids, top_k=5, temperature=0.0, mock_llm=False):
    """
    Rerank candidate_uids for a single query using a local LLM (Ollama).
    If mock_llm=True:
      - No LLM call is made.
      - Returns the first top_k candidates (excluding self).
      This is used to test pipeline without Ollama or long runtimes.
    """
    if mock_llm:
        return [u for u in candidate_uids if u != query_uid][:top_k]

    # Build candidate uid->text dict
    candidates_by_uid = {
        uid: uid_to_text[uid]
        for uid in candidate_uids
        if uid in uid_to_text and uid != query_uid
    }
    if not candidates_by_uid:
        return []

    prompt = build_rerank_prompt(query_uid, uid_to_text.get(query_uid, ""), candidates_by_uid)
    filtered = [u for u in candidate_uids if u != query_uid and u in uid_to_text] 
    out_text = ollama_generate(prompt, temperature=temperature)

# If Ollama failed/timeout, return BM25 order (filtered) so pipeline continues
    if not out_text: 
        return filtered[:top_k]


    # Parse JSON safely
    try:
        parsed = json.loads(out_text)
        ranked = parsed.get("ranked_uids", [])
        if not isinstance(ranked, list):
            ranked = []
    except Exception:
        ranked = []

    # Keep only candidate ids; pad if model omitted some
    cand_set = set(candidates_by_uid.keys())
    ranked = [uid for uid in ranked if uid in cand_set]
    if len(ranked) < top_k:
        ranked += [uid for uid in candidates_by_uid.keys() if uid not in set(ranked)]

    return ranked[:top_k]


def rerank_all(uids, raw_texts, candidates_dict, top_k=5, temperature=0.0, mock_llm=False):
    """
    Batch rerank for all query patients (loop that calls rerank_one).
    candidates_dict: uid -> list of candidate uids (ideally top 50 from BM25)
    
    Practical tip:
    - For large runs, use --sample in CLI.
    - Local CPU models will be slow for large corpora.
    """
    uid_to_text = dict(zip(uids, raw_texts))
    out = {}
    total = len(uids)

    for i, uid in enumerate(uids, start=1):
        out[uid] = rerank_one(
            uid,
            uid_to_text,
            candidates_dict.get(uid, []),
            top_k=top_k,
            temperature=temperature,
            mock_llm=mock_llm
        )

        # Print progress every 10 patients
        if not mock_llm and (i % 10 == 0 or i == total): 
            print(f"LLM reranking: processed {i}/{total}")

    return out
