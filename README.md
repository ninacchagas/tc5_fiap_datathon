# Buscador de Currículos 

## Visão Geral

Este projeto é uma aplicação que permite buscar os currículos mais aderentes a uma descrição de vaga usando embeddings de texto e transformações PCA para reduzir dimensionalidade. 
Busca apenas currículos que já tiveram prospects para agilizar o processo de deleção do candidato, isto é, mandar mensagens para quem tem maior possibilidade de passar neste processo seletivo e preencher a vaga mais rapidamente.

## Estrutura do Projeto
bash ```app/
  models/
    curriculos_emb_reduzidos.npz  # Embeddings PCA
    pca_transformer.joblib        # Transformador PCA
  app.py                # API com FastAPI
  main.py               # App Streamlit
data/
  dados_prospectados.csv # CSV com currículos prospectados
src/
  data_prep.ipynb        # Arquivo com a preparação dos dados
  modelo.py              # Modelo 
  padroniza_curriculo.py # Contém a função que realiza o tratamento dos curriculos para treinamento
Dockerfile
README.md
requirements.txt```

## Instalação

1. Clone o repositório:

    git clone <REPO_URL>
    cd <PASTA_DO_PROJETO>

2. Crie e ative um ambiente virtual:

    python -m venv .env
    source .env/bin/activate   # Linux/Mac
    .env\Scripts\activate      # Windows

3. Instale as dependências:

    pip install -r requirements.txt

## Executando o App

    streamlit run app/main.py

A aplicação abrirá no navegador na porta padrão 8501.

## Uso

1. Digite a descrição da vaga no campo de texto.
2. Clique em "Buscar currículos".
3. O app exibirá os currículos mais aderentes, mostrando:
   - E-mail
   - Telefone
   - Curriculo
   - Similaridade em %
4. É possível baixar os resultados em CSV.

## Docker

Para rodar via Docker:

1. Build da imagem:

    docker build -t buscador-curriculos .

2. Rodar o container:

    docker run -p 8501:8501 buscador-curriculos

## Observações

- O carregamento do modelo SentenceTransformer pode demorar na primeira execução.
- As embeddings e PCA já estão pré-computadas para acelerar a busca.
