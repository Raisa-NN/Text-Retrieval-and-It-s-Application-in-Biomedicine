# HI 744 Coding Assignment 2 — Patient Similarity Retrieval

This project implements patient similarity retrieval using three information retrieval approaches:

1. BM25 (traditional keyword-based retrieval)
2. Word2Vec + cosine similarity (vector-based semantic retrieval)
3. LLM-based reranking using a local Ollama model

The goal is to retrieve the most similar patient cases from a corpus of biomedical case reports and evaluate the retrieval quality using Precision@5 and Recall@5.

---

## Input Data

The program expects a directory containing multiple patient case files in JSON format (for example, the PMC_Patients dataset).
Each JSON file typically contains:

- A patient identifier (patient_uid or patient_id)
- Unstructured clinical text (e.g., title, patient)
- A similar_patients field used as gold labels for evaluation

Gold relevance labels with values 1 or 2 are treated as relevant for evaluation.

---

## What the Code Does

### Task 1 — Retrieval
For each patient in a randomly sampled subset of the corpus:

- BM25 retrieves similar patients based on shared clinical terminology
- Word2Vec represents documents as averaged word embeddings and compares them using cosine similarity
- LLM reranking uses a local large language model (via Ollama) to reorder BM25 candidates based on deeper semantic understanding

The top 5 similar patients per method are written to task1_retrieval.json.

### Task 2 — Evaluation
The retrieved results are evaluated using:

- Precision@5 — how many of the top 5 retrieved patients are truly relevant
- Recall@5 — how many of the truly relevant patients are recovered in the top 5

Results are written to task2_eval.json.

---

## Preprocessing Steps

The following text preprocessing steps are applied before retrieval:

1. Lowercasing all text
2. Tokenization using regular expressions
3. Stop-word removal (with a fallback list if NLTK stopwords are unavailable)
4. Stemming using the Porter stemmer

---

## Installation

Install dependencies using:

```bash
pip install -r requirements.txt
```

### requirements.txt

```
numpy
rank-bm25
nltk
requests
gensim
```

---

## Running the Code

From the project directory, run:

```bash
python main.py PATH_TO_INPUT_DATA_FOLDER
```

Example:

```bash
python main.py ./PMC_Patients
```

The program will:
- Randomly sample patient files
- Run BM25, Word2Vec, and LLM-based retrieval
- Save outputs to the outputs directory

---

## Using Ollama for LLM Reranking

1. Install Ollama from the official website.
2. Download a model, for example:

```bash
ollama pull mistral:7b
```

or

```bash
ollama pull llama3.1:8b
```

3. Ensure Ollama is running locally at http://localhost:11434.

LLM reranking is computationally expensive on CPU, so candidate pools are intentionally kept small to avoid timeouts.

---

## Reproducibility

Random sampling is controlled using a fixed random seed in the data-loading code to ensure reproducible results.

---

## Troubleshooting

- If input_dir errors occur, ensure the dataset path is provided.
- If gensim or nltk errors occur, confirm all dependencies are installed.
- If LLM timeouts occur, reduce candidate size or truncate text further.

---

## Notes on Performance and Limitations

- BM25 is fast but relies on exact term overlap.
- Word2Vec captures semantic similarity but may be noisy on short documents.
- LLM reranking provides deeper semantic reasoning but is slow on CPU.
- Low evaluation scores may occur when gold similar patients are not present in the sampled subset.

---

## Acknowledgments

- BM25 implemented using rank_bm25
- Word2Vec implemented using gensim
- LLM-based reranking powered by local Ollama models
