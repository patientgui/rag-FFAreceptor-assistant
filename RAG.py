import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

# 1. Charger les documents
def load_documents(file_paths):
    documents = []

    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
            documents.extend(loader.load())

    return documents


# 2. Découper les documents en chunks
def split_documents(documents, chunk_size=800, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# 3. Créer la base vectorielle
def build_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


# 4. Créer la chaîne RAG
def build_rag_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

    prompt_template = """
Tu es un assistant scientifique.
Réponds uniquement à partir du contexte fourni.
Si l'information n'est pas présente dans le contexte, dis-le clairement.
Ajoute une réponse structurée et mentionne les sources utilisées.

Contexte :
{context}

Question :
{question}

Réponse :
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

    return qa_chain


# 5. Pipeline complète
def rag_pipeline(file_paths, question):
    documents = load_documents(file_paths)
    chunks = split_documents(documents)
    vectorstore = build_vectorstore(chunks)
    qa_chain = build_rag_chain(vectorstore)

    result = qa_chain.invoke({"query": question})

    answer = result["result"]
    sources = result["source_documents"]

    return answer, sources


if __name__ == "__main__":
    files = [
        "data/article_ffa4.pdf",
        "data/notes_gpcr.docx"
    ]

    question = "Quel est le rôle du récepteur FFA4 ?"

    answer, sources = rag_pipeline(files, question)

    print("Réponse :")
    print(answer)
    print("\nSources :")
    for i, doc in enumerate(sources, 1):
        print(f"\nSource {i}:")
        print(doc.metadata)
        print(doc.page_content[:500])