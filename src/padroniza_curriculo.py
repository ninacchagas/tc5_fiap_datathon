import re
import unidecode
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('portuguese'))

def padroniza_curriculo(curriculo: str) -> str:
    curriculo = str(curriculo)
    curriculo = curriculo.lower()
    curriculo = unidecode.unidecode(curriculo)
    curriculo = re.sub(r'http\S+|www\S+|https\S+', '', curriculo, flags=re.MULTILINE)
    curriculo = re.sub(r'[^a-zA-Z\s]', ' ', curriculo)
    tokens = word_tokenize(curriculo)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    curriculo_tratado = ' '.join(tokens)
    curriculo_tratado = re.sub(r'\s+', ' ', curriculo_tratado).strip()
    return curriculo_tratado

def processa_curriculos(dados_curriculos: pd.DataFrame):
    if not dados_curriculos.empty and 'curriculo' in dados_curriculos.columns:
        curriculos_tratados = dados_curriculos['curriculo'].astype(str).apply(padroniza_curriculo)
        curriculos_a_serem_avaliados = curriculos_tratados.tolist()
    else:
        curriculos_tratados = pd.Series(dtype=str)
        curriculos_a_serem_avaliados = []
    return curriculos_tratados, curriculos_a_serem_avaliados
