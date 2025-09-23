import pandas as pd
import numpy as np
import re
import unidecode
import os
import joblib
import time

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from padroniza_curriculo import padroniza_curriculo

print("Carregando modelo Sentence Transformers...")
modelo = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("Modelo carregado.")

caminho_dados = os.path.join(os.path.dirname(__file__), '..', 'data', 'dados_prospectados.csv')
dados_curriculos = pd.read_csv(caminho_dados)

# Timer
start_time = time.time()

# Pré-processamento
if not dados_curriculos.empty and 'curriculo' in dados_curriculos.columns:
    dados_curriculos['curriculo_tratado'] = dados_curriculos['curriculo'].astype(str).apply(padroniza_curriculo)
    curriculos_a_serem_avaliados = dados_curriculos['curriculo_tratado'].tolist()
else:
    curriculos_a_serem_avaliados = []

# Embeddings
def embed_em_batches(modelo, textos, batch_size=64):
    return modelo.encode(
        textos,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True
    )

if curriculos_a_serem_avaliados:
    curriculos_emb_full = embed_em_batches(modelo, curriculos_a_serem_avaliados, batch_size=64)
    print(f"Embeddings gerados para {len(curriculos_emb_full)} currículos.")
else:
    curriculos_emb_full = np.array([])
    print("Nenhum currículo para processar.")

# PCA
if curriculos_emb_full.size > 0:
    pca = PCA(n_components=128)
    curriculos_emb = pca.fit_transform(curriculos_emb_full).astype("float16")
    print(f"Embeddings reduzidos para {curriculos_emb.shape[1]} dimensões e convertidos para float16.")
else:
    curriculos_emb = np.array([])
    print("Nenhum embedding para reduzir dimensionalidade.")

# Busca
def sentencas_similares(tema, curriculos_emb, top_n=5):
    if curriculos_emb.size == 0:
        return []

    tema_emb = modelo.encode([tema], convert_to_numpy=True)
    tema_emb = pca.transform(tema_emb).astype("float16")

    similaridades = cosine_similarity(tema_emb, curriculos_emb).flatten()
    indices_similares = np.argsort(-similaridades)

    candidatos = []
    for i in indices_similares[:top_n]:
        candidatos.append({
            "nome": dados_curriculos.iloc[i].get('nome', ''),
            "email": dados_curriculos.iloc[i].get('email', ''),
            "telefone": dados_curriculos.iloc[i].get('telefone', ''),
            "similaridade (%)": round(float(similaridades[i] * 100), 2)
        })
    return candidatos

# Salvando artefatos
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
model_dir = os.path.join(project_root, "app", "models")
os.makedirs(model_dir, exist_ok=True)

np.savez_compressed(os.path.join(model_dir, "curriculos_emb_reduzidos.npz"), curriculos_emb)
joblib.dump(pca, os.path.join(model_dir, "pca_transformer.joblib"))

print(f"Embeddings salvos em '{model_dir}'.")

# Tempo final
end_time = time.time()
print(f"Tempo de execução do script: {(end_time - start_time) / 60:.2f} minutos")
print("Processamento concluído.")