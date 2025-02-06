import os
import pandas as pd
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_csv_data(file):
    """Charge un fichier CSV et le convertit en texte."""
    file_path = os.path.join("data", file)
    if os.path.exists(file_path):
        if os.path.getsize(file_path) == 0:  # 📌 Vérifie si le fichier est vide
            print(f"⚠️ Le fichier {file} est vide, il sera ignoré.")
            return ""  # Retourne une chaîne vide pour éviter les erreurs

        try:
            df = pd.read_csv(file_path, sep=",", on_bad_lines="skip", encoding="utf-8")
        except Exception:
            df = pd.read_csv(file_path, sep=";", on_bad_lines="skip", encoding="utf-8")

        return "\n".join(df.iloc[:, 0].dropna().astype(str))  # 📌 Prend la première colonne de texte disponible
    else:
        print(f"⚠️ Fichier introuvable : {file}, il sera ignoré.")
        return ""
def chunk_text(text, chunk_size=3000, chunk_overlap=300):
    """Divise le texte en morceaux pour les embeddings."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    print("🔄 Chunking en cours...")
    chunks = list(tqdm(splitter.split_text(text), desc="Progression du chunking"))
    return chunks

def create_embeddings(chunks, batch_size=500):
    """Génère et enregistre les embeddings avec FAISS."""
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    print("🔄 Génération des embeddings...")
    vectorstore = FAISS.from_texts(chunks[:batch_size], embeddings_model)

    for i in tqdm(range(batch_size, len(chunks), batch_size), desc="Progression des embeddings"):
        vectorstore.add_texts(chunks[i:i+batch_size], embedding=embeddings_model)

    vectorstore.save_local("data/nba_knowledge")
    print("✅ Embeddings enregistrés avec succès !")

def main():
    if not os.path.exists("data"):
        raise FileNotFoundError("🚨 Le dossier data/ est introuvable.")
    
    print("📂 Chargement des données CSV...")
    csv_text = ""
    for file in ["nba_wikipedia.csv", "nba_stats.csv", "nba_website.csv", "top_20_nba_players.csv", "learned_knowledge.csv"]:
        csv_text += load_csv_data(file) + "\n"
    
    print("✂️ Chunking du texte...")
    chunks = chunk_text(csv_text)

    print("🧠 Création des embeddings...")
    create_embeddings(chunks)

    print("🎉 Préprocessing terminé avec succès !")

if __name__ == "__main__":
    main()
