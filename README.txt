# HI744 — Patient Similarity Retrieval

## Overview

This project implements patient similarity retrieval using:

1. **BM25 lexical retrieval**
2. **Local LLM reranking with Ollama**

The pipeline:
- Loads patient case JSON files
- Preprocesses text
- Retrieves BM25 top-20 candidates
- Outputs:
  - BM25 top-5 baseline
  - Reranked top-5 by an LLM
- Evaluates precision@5 and recall@5 against gold labels

## Requirements

Install Python dependencies:

```
pip install -r requirements.txt
```

Ollama (for LLM mode):
```
ollama pull mistral:7b
```

## Running the Code

### BM25 Only (optional tags for a sample and no LLM)
```
python main.py PATH_TO_INPUT_DATA_FOLDER --sample 500 --no_llm
```

### BM25 + Local LLM (Ollama) - full run of all 167k patient files
```
python main.py PATH_TO_INPUT_DATA_FOLDER
```

### Mock LLM Mode (for testing without models)
```
python main.py ./PMC_Patients --mock_llm
```

## Output Files

- `outputs/task1_retrieval.json`:  
  Contains BM25 and LLM top-5 lists per patient, and BM25 top-50 scored pool.

- `outputs/task2_eval.json`:  
  Contains per-patient precision@5 & recall@5 and dataset averages.

## Notes

- BM25 top-15 is the candidate pool for LLM reranking.
- Ollama models must be installed locally to use the `--mock_llm` flag for testing.
- BM25-only mode runs on any machine without LLMs.
- Local inference on CPU can be slow; limit `--sample` for large corpora.

## Model Recommendations

- **mistral:7b**: fast and efficient on CPU
- Adjust the model in `OLLAMA_MODEL` if needed.

## Contact

If you have questions or issues, please refer to the code or message me.

