import cohere
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction import _stop_words
import string
from config import API_KEY

co = cohere.Client(API_KEY)

def bm25_tokenizer(text):
    return [t.strip(string.punctuation) for t in text.lower().split() if t not in _stop_words.ENGLISH_STOP_WORDS]

def create_bm25(split_docs):
    tokenized_corpus = [bm25_tokenizer(doc.page_content) for doc in split_docs]
    return BM25Okapi(tokenized_corpus)

def search_with_rerank(query, bm25, split_docs, top_k=3, num_candidates=10):
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    top_n = np.argpartition(bm25_scores, -num_candidates)[-num_candidates:]
    bm25_hits = sorted([
        {'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n
    ], key=lambda x: x['score'], reverse=True)

    docs = [split_docs[hit['corpus_id']].page_content for hit in bm25_hits]
    rerank_results = co.rerank(query=query, documents=docs, top_n=top_k, return_documents=True)

    return pd.DataFrame([{
        'text': hit.document.text,
        'relevance_score': hit.relevance_score
    } for hit in rerank_results.results])