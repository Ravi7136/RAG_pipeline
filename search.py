import numpy as np
import pandas as pd

def search(query, embeddings, index, vector_store, k=3):
    query_embed = embeddings.embed_query(query)
    distances, indices = index.search(np.array([query_embed], dtype=np.float32), k)
    results = []
    for i, idx in enumerate(indices[0]):
        doc_id = vector_store.index_to_docstore_id[idx]
        doc = vector_store.docstore.search(doc_id)
        results.append({
            'texts': doc.page_content,
            'distance': distances[0][i]
        })
    return pd.DataFrame(results)