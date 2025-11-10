"""
rAg-based Policy QuEstion aNswering System

Workflow:
1. Load or create a vector database from company policy PDFs
2. Initialize the LLM and retriever
3. Build a question-answering chain
4. Query the system with policy questions
5. Display answers with source documents for transparency
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db_path = "./policy_db"
'''stored in the same directory for 
simplicity as of now '''
pdf_file = "workfromhome.pdf" 

if Path(db_path).exists():
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
else:
    
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=db_path)

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

query = "If I work remotely from another country, does it violate any company policy?"

result = qa_chain.invoke({"query": query})

print(result["result"])

for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n[Document {i}]")

