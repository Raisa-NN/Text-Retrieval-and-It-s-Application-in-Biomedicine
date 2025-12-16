# -*- coding: utf-8 -*-
"""
This module is designed for pipelines that store one patient per JSON file.
It provides:
  - `read_patient_jsons(...)` to load and extract patient identifiers + text
  - `save_json(...)` to write results (e.g., retrieval outputs) as JSON

@author: raisa
"""

# io_utils.py
import os
import json
import glob
import ast


def read_patient_jsons(input_dir, limit=None):
    """
    Reads all `*.json` files in `input_dir`.

    Returns:
        patient_uids (list[str]):
            Unique identifier for each patient.
        raw_texts (list[str]):
            Raw patient text assembled from the JSON fields.
        gold_similar (dict[str, list[str]]):
            Gold-standard similar patients for evaluation.
            Each key is a patient_uid, and each value is a list of
            relevant patient_uids (treating relevance levels 1 and 2 equally).

    Notes:
        - The `similar_patients` field in the JSON may be stored as:
            * a dictionary
            * a string representation of a dictionary
        - Relevance values (1 or 2) are ignored; only the keys are used,
          per assignment instructions.
    """
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if limit is not None:
        files = files[:limit]

    uids = []
    texts = []
    gold = {}  # uid -> list of similar uids

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue  # skip unreadable files

        # --------------------
        # Patient UID
        # --------------------
        uid = str(
            data.get("patient_uid")
            or data.get("patient_id")
            or os.path.basename(fp) # extracts the filename only and ignores path
        )

        # --------------------
        # Patient text
        # --------------------
        parts = []
        if isinstance(data.get("title"), str):
            parts.append(data["title"])
        if isinstance(data.get("patient"), str):
            parts.append(data["patient"])

        # Fallback: include all string fields if title/patient missing
        if not parts:
            for v in data.values():
                if isinstance(v, str):
                    parts.append(v)

        texts.append("\n".join(parts))
        uids.append(uid)

        # --------------------
        # Gold similar patients
        # --------------------
        sp = data.get("similar_patients", {})
        sp_dict = {}

        if isinstance(sp, dict):
            sp_dict = sp
        elif isinstance(sp, str):
            try:
                sp_dict = ast.literal_eval(sp)
                # safely converts a string that looks like a Python literal (dict/list/tuple/str/num) into the actual Python object.
            except Exception:
                sp_dict = {}

        # Treat relevance levels 1 and 2 equally (binary relevance)
        gold[uid] = [str(k) for k in sp_dict.keys()]

    return uids, texts, gold

def save_json(obj, path):
    # Saves a Python object to a JSON file (pretty-printed, UTF-8).
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)