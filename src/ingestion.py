import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

def ingest_docs():
    print("--- Début de l'ingestion des documents ---")
    
    # 1. Chargement des PDF depuis le dossier data
    loader = DirectoryLoader('data/', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"Nombre de pages chargées : {len(documents)}")
    
    # 2. Découpage en morceaux (Chunks)
    # On utilise 800 caractères avec un recouvrement pour garder le contexte médical
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(f"Nombre de fragments créés : {len(docs)}")
    
    # 3. Création des embeddings et stockage
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    PineconeVectorStore.from_documents(
        docs, 
        embeddings, 
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    
    print("--- Ingestion terminée avec succès dans Pinecone ! ---")

if __name__ == "__main__":
    ingest_docs()