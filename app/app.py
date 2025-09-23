from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os
import joblib
import pandas as pd

base_path = os.path.dirname(os.path.abspath(__file__))
curriculos_emb_path = os.path.join(base_path, "models", "curriculos_emb_reduzidos.npz")
pca_path = os.path.join(base_path, "models", "pca_transformer.joblib")
dados_curriculos_path = os.path.join(base_path, "..", "data", "dados_prospectados.csv")

for path in [curriculos_emb_path, pca_path, dados_curriculos_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

curriculos_emb = np.load(curriculos_emb_path)['arr_0']
pca = joblib.load(pca_path)
df_curriculos = pd.read_csv(dados_curriculos_path)
modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title="API Recrutamento")

class Vaga(BaseModel):
    descricao: str
    top_n: int = 5

def busca_candidatos(tema, top_n=5):
    if curriculos_emb.size == 0:
        return []

    tema_emb = modelo.encode([tema], convert_to_numpy=True)
    tema_emb = pca.transform(tema_emb).astype("float16")

    similaridades = cosine_similarity(tema_emb, curriculos_emb).flatten()
    indices_similares = np.argsort(-similaridades)

    candidatos = []
    for i in indices_similares[:top_n]:
        candidatos.append({
            "indice_curriculo": int(i),
            "nome": df_curriculos.iloc[i].get('nome', ''),
            "email": df_curriculos.iloc[i].get('email', ''),
            "telefone": df_curriculos.iloc[i].get('telefone', ''),
            # "curriculo_texto": df_curriculos.iloc[i].get('curriculo', ''),
            "similaridade (%)": round(float(similaridades[i] * 100), 2)
        })
    return candidatos

@app.post("/buscar_candidatos")
def buscar_candidatos_endpoint(vaga: Vaga):
    return {"top_candidatos": busca_candidatos(vaga.descricao, vaga.top_n)}