import os
import streamlit as st
from src.chat_engine import get_response

st.set_page_config(page_title="Syfl Medical Chatbot", page_icon="💊")

# Style personnalisé
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    .stChatMessage { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

st.title("💊 Assistant Médical IA")
st.caption("Base de connaissances basée sur vos documents officiels")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Affichage des anciens messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrée utilisateur
if prompt := st.chat_input("Posez votre question médicale ici..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyse des sources en cours..."):
            result = get_response(prompt)
            answer = result["result"]
            sources = result["source_documents"]
            
            st.markdown(answer)
            
            # Affichage des sources de manière propre
            if sources:
                with st.expander("Sources consultées"):
                    for i, doc in enumerate(sources):
                        source_name = os.path.basename(doc.metadata.get('source', 'Inconnu'))
                        st.write(f"Source {i+1} : {source_name} (Page {doc.metadata.get('page', '?')})")

    st.session_state.messages.append({"role": "assistant", "content": answer})