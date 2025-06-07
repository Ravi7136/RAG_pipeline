from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
index = faiss.IndexFlatIP(384)

def create_vector_store(split_docs):
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(split_docs)
    return vector_store, index, embeddings