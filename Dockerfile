FROM python:3.12-slim

WORKDIR /project_root

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    unixodbc-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
