from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(pages):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(pages)