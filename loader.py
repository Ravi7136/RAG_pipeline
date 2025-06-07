from langchain_community.document_loaders import PyPDFLoader
import asyncio

async def load_pdf_pages(path):
    loader = PyPDFLoader(path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages