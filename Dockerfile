FROM python:3.12-slim

# Define o diretório de trabalho dentro do container
WORKDIR /tc5_fiap_datathon

# Instala dependências do sistema necessárias para algumas libs do Python
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Atualiza pip
RUN pip install --upgrade pip

# Copia o arquivo de dependências e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código para dentro do container
COPY . .

# Expõe a porta da API
EXPOSE 8000

# Comando para rodar a API FastAPI
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
