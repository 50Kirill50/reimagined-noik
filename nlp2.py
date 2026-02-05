# 2 Кластеризация
"""
Суть: Разделить данные на K кластеров так, чтобы объекты внутри кластера были похожи.
Как работает:

Случайно выбрать K центров кластеров
Назначить каждый объект ближайшему центру
Пересчитать центры (среднее по объектам в кластере)
Повторять шаги 2-3, пока центры не стабилизируются
"""

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

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

df = pd.read_excel(r"TextsUnlabeledVar1.xlsx", header=None)
df.columns = ["text"]
df = df.dropna()
df['text_clean'] = df['text'].apply(preprocess)  # Ваша функция
df

# ============ ВЕКТОРИЗАЦИЯ ============
# vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
vectorizer = TfidfVectorizer(
    max_features=5000,    # Ограничить число признаков
    ngram_range=(1, 2),   # Униграммы + биграммы
    min_df=2,             # Игнорировать редкие слова
    max_df=0.8            # Игнорировать слишком частые
)
X = vectorizer.fit_transform(df['text_clean'])

feature_names = vectorizer.get_feature_names_out()
    
# Sum TF-IDF scores across all documents
tfidf_sums = np.asarray(X.sum(axis=0)).flatten()
    
# Get indices of top N words
top_indices = np.argsort(tfidf_sums)[::-1][:10]
    
# Return top words and their scores
top_words = [(feature_names[i], tfidf_sums[i]) for i in top_indices]

# Get top words across all document
for word, score in top_words:
    print(f"{word}: {score:.4f}")

# ============ КЛАСТЕРИЗАЦИЯ ============
kmeans = KMeans(
    n_clusters=2,      # Число кластеров
    random_state=42,   # Для воспроизводимости
    n_init=10,         # Число попыток с разными начальными центрами
    max_iter=300       # Максимум итераций
)

clusters = kmeans.fit_predict(X)
df['cluster'] = clusters

# ============ АНАЛИЗ КЛАСТЕРОВ ============
print("Размеры кластеров:")
print(df['cluster'].value_counts())

# Посмотреть ключевые слова каждого кластера
feature_names = vectorizer.get_feature_names_out()

for i in range(2):
    print(f"\n=== КЛАСТЕР {i} ===")
    
    # Центр кластера (средний TF-IDF вектор)
    cluster_center = kmeans.cluster_centers_[i]
    
    # Топ-15 слов с наибольшим весом
    top_indices = cluster_center.argsort()[-15:][::-1]
    top_words = [feature_names[idx] for idx in top_indices]
    
    print(f"Ключевые слова: {', '.join(top_words)}")
    print(f"Размер кластера: {(clusters == i).sum()}")

# ============ ОПРЕДЕЛЕНИЕ, ЧТО ЕСТЬ ЧТО ============
# Ручной анализ: смотрим на ключевые слова
# Если кластер 0: "президент", "страна", "правительство" → новости
# Если кластер 1: "отель", "номер", "персонал" → отзывы

review_cluster = 0  # Определяем вручную!

# ОТВЕТ
answer = (clusters == review_cluster).sum()
print(f"\nОтвет: {answer}")

# Если есть несколько примеров отзывов, можно автоматизировать
review_keywords = ['отель', 'номер', 'персонал', 'сервис', 'завтрак']

# Для каждого кластера подсчитать, сколько ключевых слов отзывов
cluster_scores = []
for i in range(2):
    cluster_center = kmeans.cluster_centers_[i]
    
    # Суммарный вес ключевых слов отзывов
    score = 0
    for word in review_keywords:
        if word in feature_names:
            word_idx = np.where(feature_names == word)[0][0]
            score += cluster_center[word_idx]
    
    cluster_scores.append(score)

review_cluster = np.argmax(cluster_scores)
print(f"Кластер отзывов: {review_cluster}")
