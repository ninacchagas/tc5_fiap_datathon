# imagem base com Python 3.12
FROM python:3.12-slim

# define diretório de trabalho
WORKDIR /app

# copia requirements se houver
COPY requirements.txt .

# instala dependências
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# copia o restante do código
COPY . .

# expõe a porta do Streamlit
EXPOSE 8501

# define variáveis de ambiente para rodar Streamlit sem warnings
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false \
    STREAMLIT_SERVER_PORT=8501

# comando para iniciar o app
CMD ["streamlit", "run", "app/main.py"]
