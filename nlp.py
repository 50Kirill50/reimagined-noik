"""
Сценарии применения TF-IDF.
1. Мера TF-IDF позволяет выделить ключевые слова в каждом из документов. Для этого надо выбрать те слова, которые дают наибольшее значение меры TF-IDF для текущего документа.
Извлечение ключевых слов с помощью TF-IDF полезно, например, для индексации текстовых данных и ранжирования результатов поисковых запросов по коллекции документов.
2. Мера TF-IDF может быть использована для кластеризации текстовых документов. Это позволит сгруппировать схожие документы в один кластер, выявив общие темы.
Например, в социальных сетях можно кластеризовать посты пользователей для создания персонализированных лент новостей.
3. Меру TF-IDF можно также использовать для задачи классификации текстов по их содержанию. Для этого надо обучить модель машинного обучения на векторах, полученных после работы метода TF-IDF на обучающем наборе документов. А затем можно использовать эту модель для предсказания класса, к которому принадлежат новые тексты.
Например, это можно использовать для решения задачи фильтрации спама в электронных письмах.
"""

import re
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import nltk
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

# ============ БАЗОВОЕ ИСПОЛЬЗОВАНИЕ ============
texts = [
    "хорош отел персонал вежлив",
    "плох отел грязн номер",
    "отличн ресторан вкусн еда"
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# X - это разреженная матрица (sparse matrix)
print(X.shape)  # (3, 10) - 3 документа, 10 уникальных слов
print(X.toarray())  # Превратить в обычный numpy array

# Посмотреть словарь (какие слова на каких позициях)
print(vectorizer.get_feature_names_out())

df = pd.read_csv('reviews.csv')

# 1. Предобработать тексты
df['text_clean'] = df['text'].apply(preprocess)

# 2. Векторизация
vectorizer = TfidfVectorizer(
    max_features=5000,      # Взять топ-5000 слов по частоте
    min_df=2,               # Игнорировать слова, которые встречаются < 2 раз
    max_df=0.8,             # Игнорировать слова, которые встречаются > 80% документов
    ngram_range=(1, 2)      # Учитывать биграммы (пары слов)
)

X = vectorizer.fit_transform(df['text_clean'])

# 3. Использовать для классификации
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = train_test_split(X, df['label'], test_size=0.2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
