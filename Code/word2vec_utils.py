# -*- coding: utf-8 -*-

import numpy as np
try:
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
except ImportError:
    Word2Vec = None
    GENSIM_AVAILABLE = False


def train_word2vec(tokenized_docs):
    """
    Train a Word2Vec model on the tokenized corpus.

    Word2Vec learns vector representations for words such that
    semantically similar words have similar vectors.
    """
    model = Word2Vec(
        sentences=tokenized_docs,
        vector_size=200, # embedding dimension
        window=5, # context window
        min_count=2, # ignore very rare words
        workers=4, # CPU threads
        sg=1, # skip-gram model; predicts the context words given the center word.
        epochs=10,
        seed=42
    )
    return model.wv


def document_vectors(tokenized_docs, wv):
    """
    Convert each document into a single vector by averaging its word vectors.
    """
    vectors = []

    for tokens in tokenized_docs:
        word_vecs = [wv[t] for t in tokens if t in wv]

        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(wv.vector_size))

    return np.vstack(vectors)
    # np.vstack stacks a list of 1D vectors into a 2D array:
    # Converts a list of vectors into a matrix


def cosine_similarity_matrix(X):
    """
    Compute cosine similarity between all document vectors.

    np.linalg.norm computes vector length (L2 norm).
    Dividing by the norm normalizes vectors to unit length,
    allowing dot product to equal cosine similarity.
    """
    # computes the length (magnitude) of each row vector
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    # rescales each row so it has length 1 (a “unit vector”)
    # 1e-12 is just a tiny constant to avoid division by zero if a vector is all zeros.
    X_norm = X / (norms + 1e-12)
    # multiplies the normalized matrix by its transpose.
    return X_norm @ X_norm.T


def word2vec_retrieve(processed_docs, uids, top_k=5):
    """
    Retrieve similar patients using Word2Vec document embeddings.
    If gensim is not available, returns empty results and prints a warning.
    """
    if not GENSIM_AVAILABLE:
        print("WARNING: gensim not installed. Skipping Word2Vec retrieval.")
        return {uid: [] for uid in uids}

    
    tokenized = [d.split() for d in processed_docs]
    
    # Train a Word2Vec model and get access to the learned word vectors
    # 'wv' maps: word -> numeric vector (embedding)
    wv = train_word2vec(tokenized)
    doc_vecs = document_vectors(tokenized, wv)
    sims = cosine_similarity_matrix(doc_vecs)

    results = {}
    for i, uid in enumerate(uids):
        scores = sims[i]
        
        # Build (candidate_uid, similarity_score) pairs
        # Exclude the document itself (no self-retrieval)
        pairs = [
            (uids[j], scores[j])
            for j in range(len(uids))
            if uids[j] != uid
        ]

        pairs.sort(key=lambda x: x[1], reverse=True)
        results[uid] = [u for u, _ in pairs[:top_k]]

    return results


