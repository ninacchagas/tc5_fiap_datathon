# main.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ----------------------------------------------------
# Diretórios e arquivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # pasta 'app'
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
CSV_FILE = "dados_prospectados.csv"
CSV_PATH = os.path.abspath(os.path.join(DATA_DIR, CSV_FILE))

# ----------------------------------------------------
# Carrega currículos, embeddings e modelo Sentence Transformers
@st.cache_resource
def load_models():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Arquivo CSV não encontrado: {CSV_PATH}")
    if not os.path.exists(os.path.join(MODEL_DIR, "curriculos_emb_reduzidos.npz")):
        raise FileNotFoundError("Arquivo curriculos_emb_reduzidos.npz não encontrado")
    if not os.path.exists(os.path.join(MODEL_DIR, "pca_transformer.joblib")):
        raise FileNotFoundError("Arquivo pca_transformer.joblib não encontrado")

    df_curriculos = pd.read_csv(CSV_PATH)
    curriculos_embeddings = np.load(os.path.join(MODEL_DIR, "curriculos_emb_reduzidos.npz"))['arr_0']
    pca = joblib.load(os.path.join(MODEL_DIR, "pca_transformer.joblib"))
    modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return df_curriculos, curriculos_embeddings, pca, modelo

df_curriculos, curriculos_embeddings, pca, modelo = load_models()

# ----------------------------------------------------
# Função para buscar currículos mais aderentes
def sentencas_similares(tema, df_curriculos, curriculos_emb, modelo, pca, top_n=5):
    if curriculos_emb.size == 0:
        return pd.DataFrame()

    # Gera embedding da query e aplica PCA
    tema_emb = modelo.encode([tema], convert_to_numpy=True)
    tema_emb = pca.transform(tema_emb).astype("float16")

    # Calcula similaridade
    similaridades = cosine_similarity(tema_emb, curriculos_emb).flatten()
    indices_similares = np.argsort(-similaridades)[:top_n]

    resultados = []
    for i in indices_similares:
        resultados.append({
            # "nome": df_curriculos.iloc[i].get('nome', ''),
            "email": df_curriculos.iloc[i].get('email', ''),
            "telefone": df_curriculos.iloc[i].get('telefone', ''),
            "curriculo_texto": df_curriculos.iloc[i].get('curriculo', ''),
            # "nivel_academico": df_curriculos.iloc[i].get('nivel_academico', ''),
            "similaridade (%)": round(float(similaridades[i]*100), 2)
        })
    return pd.DataFrame(resultados)

# ----------------------------------------------------
# Interface Streamlit
st.title("Buscador de Currículos")
st.write("Digite a descrição da vaga e encontre os currículos mais aderentes.")

descricao_vaga = st.text_area("Descrição da vaga")

if st.button("Buscar currículos"):
    if descricao_vaga.strip() == "":
        st.warning("Digite a descrição da vaga antes de buscar.")
    else:
        resultados = sentencas_similares(descricao_vaga, df_curriculos, curriculos_embeddings, modelo, pca)
        if resultados.empty:
            st.info("Nenhum currículo encontrado.")
        else:
            st.success("Currículos encontrados:")
            st.dataframe(resultados)

            csv = resultados.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Baixar resultados em CSV",
                data=csv,
                file_name="curriculos_resultado.csv",
                mime="text/csv"
            )
