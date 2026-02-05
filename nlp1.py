from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
import re
import string
import pandas as pd

# ============ СКАЧАТЬ NLTK ДАННЫЕ (ОДИН РАЗ!) ============
# nltk.download('stopwords')
# nltk.download('punkt')

stop_words = set(stopwords.words('russian'))
stemmer = SnowballStemmer('russian')

def preprocess(text: str):
    # Если что, не
    # 1. Приведение текста к нижнему регистру
    text = text.lower()
    # 2. Удаление чисел (или приведение их к другому формату)
    text = re.sub(r'\d+', '', text)
    # ИЛИ
    # text = re.sub(r'\d+', '<NUM>', text)
    # text = re.sub(r'\d+\.?\d*', '<NUM>', text)
    # 3. Удаление знаков препинания
    text = ''.join([c for c in text if c not in string.punctuation])
    # 4. Токенизация: разбиение длинных строк на более короткие (например, разбиение по словам)
    words = text.split()  # Простая токенизация
    # words = word_tokenize(text)  # Умная токенизация (медленнее)
    # 5. Удаление стоп-слов, то есть слов, которые не несут смысловой нагрузки (союзы, предлоги)
    words = [w for w in words if w not in stop_words]
    # 6. Стемминг: приведение слов к основной форме (обычно означает удаление окончаний слов)
    words = [stemmer.stem(w) for w in words]
    return " ".join(words)

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np


# CSV
df = pd.read_excel(r"TextsUnlabeledVar1.xlsx", header=None)
df.columns = ["text"]
df = df.dropna()

df = df[df["text"].str.len() > df["text"].str.len().max()//10].copy()

df['text_clean'] = df['text'].apply(preprocess)  # Ваша функция

df

# ============ ВЕКТОРИЗАЦИЯ ============

"""
# ============ ТОЛЬКО УНИГРАММЫ ============
vectorizer = TfidfVectorizer(ngram_range=(1, 1))
# Слова: ['отель', 'хороший', 'персонал']

# ============ УНИГРАММЫ + БИГРАММЫ ============
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
# Слова: ['отель', 'хороший', 'персонал', 'хороший отель', 'хороший персонал']

# ============ ТОЛЬКО БИГРАММЫ ============
vectorizer = TfidfVectorizer(ngram_range=(2, 2))
# Слова: ['хороший отель', 'хороший персонал']

# ============ УНИГРАММЫ + БИГРАММЫ + ТРИГРАММЫ ============
vectorizer = TfidfVectorizer(ngram_range=(1, 3))
# Много признаков! Может переобучиться
"""

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text_clean'])

# ============ ИЗВЛЕЧЕНИЕ КЛЮЧЕВЫХ СЛОВ ============
# doc_id = 42
# doc_idx = df[df['id'] == doc_id].index[0]

needed_doc = df.iloc[123] # 2
# Получить TF-IDF веса для этого документа
tfidf_scores = X[123].toarray().flatten()
tfidf_scores

# Получить названия слов
feature_names = vectorizer.get_feature_names_out()
feature_names

"""
"Найдите 5 ключевых слов документа X"
"Какое слово имеет наибольший TF-IDF в тексте Y?"
"""

# Создать DataFrame для удобства
keywords_df = pd.DataFrame({
    'word': feature_names,
    'tfidf': tfidf_scores
}).sort_values('tfidf', ascending=False)

# Топ-10 ключевых слов
top_keywords = keywords_df[keywords_df['tfidf'] > 0].head(10)
print(top_keywords)

# ОТВЕТ (если нужны только слова)
answer = ', '.join(top_keywords['word'].tolist())
print(f"Ответ: {answer}")
