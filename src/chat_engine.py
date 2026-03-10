import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def get_response(query):
    # 1. Configuration des Embeddings et du VectorStore
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"), 
        embedding=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 2. Configuration du Modèle
    llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

    # 3. Création du Prompt
    template = """Tu es un assistant médical expert. Réponds à la question en te basant 
    uniquement sur le contexte suivant :
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # 4. Construction de la "Chain" avec LCEL (plus stable en 2026)
    # Cette syntaxe évite les imports de "langchain.chains"
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 5. Exécution
    response_text = rag_chain.invoke(query)
    
    # On récupère aussi les docs pour l'affichage des sources dans app.py
    docs = retriever.invoke(query)

    return {
        "result": response_text,
        "source_documents": docs
    }