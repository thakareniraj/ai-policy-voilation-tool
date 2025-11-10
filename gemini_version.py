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
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA


os.environ["GOOGLE_API_KEY"] = "AIzaSyDNMlFGB_vZHbl9AcI8FpzoOeGXTGY3_rc"

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

db_path = "./policy_db"
pdf_file = "workfromhome.pdf"

if Path(db_path).exists():
    db = Chroma(persist_directory=db_path, embedding_function=embeddings)
    print(f"Successfully load database from {db_path}")
else:
    print("creating new db")
    
    loader = PyPDFLoader(pdf_file)
    documents = loader.load()
    
    print("Splitting step")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    db = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=db_path)
    print("created a saved vector db")

retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

query = "If I work remotely from another country, does it violate any company policy?"

result = qa_chain.invoke({"query": query})

print("ANSWER:")
print(result["result"])
print()

print("SOURCE DOCUMENTS USED FOR THIS ANSWER:")
for i, doc in enumerate(result["source_documents"], 1):
    print(f"\n[Document {i}]")
    content_preview = doc.page_content[:500]
    print(content_preview)
    if len(doc.page_content) > 500:
        print("...")
    print()
