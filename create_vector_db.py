"""
create_vector_db.py
Indexa múltiplas fontes PDF em um único vector store FAISS.
Execute uma vez (ou quando adicionar novas fontes):
    python create_vector_db.py
"""

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# ─────────────────────────────────────────
# FONTES — adicione seus PDFs aqui
# prioridade: 1 = máxima autoridade, 2 = validação, 3 = específico
# ─────────────────────────────────────────
FONTES = [
    # ── Clínicas Primárias (Young, Klosko — base teórica da TE) ──────────
    {
        "arquivo": "fontes/reinvente_sua_vida.pdf",
        "prioridade": 1,
        "tipo": "clinica_primaria",
        "descricao": "Young & Klosko — Reinvente Sua Vida (versão popular da TE)"
    },
    
    {
        "arquivo": "fontes/guia_de_tecnicas.pdf",
        "prioridade": 1,
        "tipo": "clinica_primaria",
        "descricao": "Young, Klosko & Weishaar — Schema Therapy: A Practitioner's Guide"
    },

    # ── Artigos Científicos (validação empírica) ──────────────────────────
     {
         "arquivo": "fontes/new_look_schema.pdf",
         "prioridade": 2,
         "tipo": "artigo_cientifico",
         "descricao": "Bach et al. (2018) — A New Look at the Schema Therapy Model"
     },
    {
         "arquivo": "fontes/schema_modes.pdf",
         "prioridade": 2,
         "tipo": "artigo_cientifico",
         "descricao": "Edwards (2021) — Using Schema Modes for Case Conceptualization"
     },
]

# ─────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = "schema_db"

def main():
    print("🔧 Iniciando indexação...\n")

    # Carrega modelo de embeddings (faz download na primeira vez)
    print(f"📦 Carregando modelo de embeddings: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    all_chunks = []
    fontes_processadas = 0

    for fonte in FONTES:
        arquivo = fonte["arquivo"]

        if not os.path.exists(arquivo):
            print(f"⚠️  '{arquivo}' não encontrado — pulando.")
            continue

        print(f"📄 Processando: {arquivo}")
        try:
            docs = PyPDFLoader(arquivo).load()
            chunks = splitter.split_documents(docs)

            # Adiciona metadados de hierarquia em cada chunk
            for chunk in chunks:
                chunk.metadata["fonte"] = arquivo
                chunk.metadata["prioridade"] = fonte["prioridade"]
                chunk.metadata["tipo"] = fonte["tipo"]
                chunk.metadata["descricao"] = fonte.get("descricao", "")

            all_chunks.extend(chunks)
            fontes_processadas += 1
            print(f"   ✅ {len(chunks)} chunks gerados")

        except Exception as e:
            print(f"   ❌ Erro ao processar {arquivo}: {e}")

    if not all_chunks:
        print("\n❌ Nenhum chunk gerado. Verifique se os PDFs estão na pasta correta.")
        return

    # Cria e salva o vector store
    print(f"\n🧠 Gerando embeddings para {len(all_chunks)} chunks...")
    db = FAISS.from_documents(all_chunks, embeddings)
    db.save_local(OUTPUT_DIR)

    print(f"\n🎉 Concluído!")
    print(f"   Fontes processadas : {fontes_processadas}")
    print(f"   Total de chunks    : {len(all_chunks)}")
    print(f"   Índice salvo em    : {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()