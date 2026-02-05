# 3 Классификация
# Обучить модель различать новости от отзывов.


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

# stop_words = set(stopwords.words('russian'))
# stemmer = SnowballStemmer('russian')

stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

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

from sklearn.datasets import fetch_20newsgroups

categories = ['comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

X = newsgroups.data
y = newsgroups.target
target_names = newsgroups.target_names

X

X = pd.Series(X)
X = X.apply(lambda x: preprocess(x))
X

# ============ ВЕКТОРИЗАЦИЯ ============
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_vc = vectorizer.fit_transform(X)
X_vc

# ============ РАЗДЕЛЕНИЕ ============
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_vc, y, test_size=0.2, stratify=y, random_state=42
)

from sklearn.linear_model import LogisticRegression
# ============ ОБУЧЕНИЕ ============
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# ============ ОЦЕНКА ============
from sklearn.metrics import accuracy_score, classification_report
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(classification_report(y_test, predictions))

# ============ АНАЛИЗ ВАЖНЫХ СЛОВ ============
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]
coefficients

X[1], y[1]

# Слова, указывающие на медицину (положительные коэффициенты)
review_words = pd.DataFrame({
    'word': feature_names,
    'coef': coefficients
}).sort_values('coef', ascending=False).head(20)

print("\nСлова, указывающие на медицину:")
print(review_words)

# Слова, указывающие на IT (отрицательные коэффициенты)
news_words = pd.DataFrame({
    'word': feature_names,
    'coef': coefficients
}).sort_values('coef', ascending=True).head(20)

print("\nСлова, указывающие на IT:")
print(news_words)

y.sum()

# ОТВЕТ (для всех данных)
all_predictions = model.predict(X_vc)
answer = (all_predictions == 1).sum()
print(f"\nОтвет: {answer}")
