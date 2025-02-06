import os
import ollama
import streamlit as st
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def load_vectorstore():
    """Charge les embeddings FAISS."""
    if not os.path.exists("data/nba_knowledge"):
        raise FileNotFoundError("🚨 Embeddings FAISS introuvables. Exécute `preprocessing.py` d'abord !")

    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    return FAISS.load_local("data/nba_knowledge", embeddings_model, allow_dangerous_deserialization=True)

def save_new_knowledge(query, response):
    """Sauvegarde les nouvelles réponses générées dans un fichier CSV."""
    file_path = "data/learned_knowledge.csv"
    
    # Vérifie si le fichier existe, sinon le crée avec un en-tête
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("Question,Réponse\n")

    # Ajoute la nouvelle question-réponse
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"\"{query}\",\"{response}\"\n")

    print("💾 Nouvelle connaissance ajoutée !")

def generate_response_with_ollama(query, retrieved_texts):
    """Utilise Ollama pour générer une réponse en français."""
    if not retrieved_texts:
        return "Je n'ai trouvé aucune information pertinente pour répondre à cette question."

    context = "\n\n".join(retrieved_texts)
    prompt = f"""
    Contexte :
    {context}

    Question : {query}
    Réponds de manière claire et concise en français en t'appuyant uniquement sur les informations ci-dessus.
    """

    try:
        response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
        response_text = response["message"]["content"]
        save_new_knowledge(query, response_text)  # Sauvegarde la nouvelle connaissance
        return response_text
    except Exception as e:
        return f"⚠️ Erreur lors de l'appel à Ollama : {e}"

def search_query(query, top_k=3):
    """Recherche les documents les plus pertinents via FAISS."""
    try:
        vectorstore = load_vectorstore()
        results = vectorstore.similarity_search_with_score(query, k=top_k)
        
        # Prioriser les nouvelles connaissances
        results = sorted(results, key=lambda x: ("learned_knowledge.csv" in x[0].metadata.get("source", ""), x[1]), reverse=True)
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la recherche : {e}")
        return []

    return [doc.page_content for doc, score in results if doc.page_content.strip()]

def main():
    st.title("🏀 B-BALL")
    st.write("Pose une question sur la NBA et obtiens une réponse intelligente qui s'améliore avec le temps !")
    
    query = st.text_input("🔎 Pose ta question sur la NBA :")
    
    if st.button("Envoyer"):
        if not query:
            st.warning("⚠️ Veuillez entrer une question valide !")
        else:
            results = search_query(query)
            retrieved_texts = [content for content in results]
            final_response = generate_response_with_ollama(query, retrieved_texts)
            
            st.subheader("🎯 Réponse générée :")
            st.write(final_response)

if __name__ == "__main__":
    main()
