from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

prompt = PromptTemplate(
    template="""<|user|>
Relevant information:
{context}

Provide a concise answer the following question using the relevant information provided above:
{question}<|end|>
<|assistant|>""",
    input_variables=["context", "question"]
)

model = ChatGoogleGenerativeAI(model='gemini-1.5-flash')

def build_qa_pipeline(vector_store):
    return RetrievalQA.from_chain_type(
        llm=model,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        chain_type_kwargs={"prompt": prompt},
        verbose=True
    )