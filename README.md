# 🧠 Thesis Research Assistant

Assistant intelligent basé sur le **RAG (Retrieval-Augmented Generation)** pour interroger un corpus scientifique local (articles, notes, documents de thèse, bibliographie) avec un **LLM local** via Ollama.

Le projet permet de :
- répondre à des questions à partir du corpus documentaire
- retrouver des documents pertinents à partir d’une requête
- générer des résumés
- générer des quiz de révision
- effectuer une recherche web en complément

---

## ✨ Fonctionnalités

### 1. Question-réponse sur corpus
L’assistant recherche les passages les plus pertinents dans les documents puis génère une réponse structurée.

### 2. Recherche de documents
À partir d’une requête, l’assistant retrouve les documents ou passages les plus pertinents.

### 3. Résumé
L’assistant peut produire un résumé clair d’un thème ou d’un ensemble de documents.

### 4. Quiz
L’assistant peut générer un quiz de révision à partir du corpus.

### 5. Recherche web
L’assistant peut compléter les documents locaux avec des résultats issus du web.

---

## 🏗️ Architecture du projet

```bash
RAG-FFA4/
├── app.py                  # point d’entrée principal
├── rag.py                  # pipeline RAG
├── tools.py                # outils (web, résumé, quiz, recherche doc)
├── router.py               # routage des requêtes
├── premium_streamlit_thesis_assistant.py   # interface Streamlit premium
├── data/                   # corpus documentaire (PDF / DOCX)
├── faiss_index/            # index vectoriel FAISS
├── requirements.txt
├── .gitignore
└── README.md
