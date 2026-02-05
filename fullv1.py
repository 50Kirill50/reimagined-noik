"""
СОДЕРЖАНИЕ:
1. ПОДГОТОВКА (NLTK, импорты)
2. NLP + TF-IDF (3 сценария)
3. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
4. ПАРАМЕТРИЧЕСКОЕ МОДЕЛИРОВАНИЕ
5. PANDAS ШПАРГАЛКА (для тестовых заданий)
"""

# ═══════════════════════════════════════════════════════════════════════════
# 1. ПОДГОТОВКА
# ═══════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import re
import string
from scipy.optimize import minimize, curve_fit
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# ═══ СКАЧАТЬ NLTK ДАННЫЕ (ОДИН РАЗ) ═══
# nltk.download('stopwords')
# nltk.download('punkt')

# ═══════════════════════════════════════════════════════════════════════════
# 2. NLP + TF-IDF
# ═══════════════════════════════════════════════════════════════════════════

# ═══ ПРЕДОБРАБОТКА ТЕКСТА ═══
def preprocess_text(text, language='russian', 
                    remove_numbers=True, 
                    remove_stopwords=True, 
                    stem=True):
    """
    Полная предобработка текста
    
    Параметры:
    - language: 'russian' или 'english'
    - remove_numbers: удалять числа?
    - remove_stopwords: удалять стоп-слова?
    - stem: применять стемминг?
    """
    stop_words = set(stopwords.words(language))
    stemmer = SnowballStemmer(language)
    
    # 1. Нижний регистр
    text = text.lower()
    
    # 2. Удалить числа
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    # 3. Удалить пунктуацию
    text = ''.join([c for c in text if c not in string.punctuation])
    
    # 4. Токенизация (простая)
    words = text.split()
    
    # 5. Удалить стоп-слова
    if remove_stopwords:
        words = [w for w in words if w not in stop_words]
    
    # 6. Стемминг
    if stem:
        words = [stemmer.stem(w) for w in words]
    
    return " ".join(words)


# ═══════════════════════════════════════════════════════════════════════════
# СЦЕНАРИЙ 1: КЛАССИФИКАЦИЯ ТЕКСТОВ (новости vs отзывы)
# ═══════════════════════════════════════════════════════════════════════════
def nlp_classification(csv_file, text_column, label_column=None):
    """
    Классификация текстов на 2 класса
    
    Параметры:
    - csv_file: путь к файлу
    - text_column: название колонки с текстом
    - label_column: название колонки с метками (если есть)
    
    Возвращает: количество текстов класса 1
    """
    # Загрузка
    df = pd.read_csv(csv_file)
    
    # Предобработка
    df['text_clean'] = df[text_column].apply(preprocess_text)
    
    # Векторизация
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # униграммы + биграммы
        min_df=2,
        max_df=0.8
    )
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
    X = vectorizer.fit_transform(df['text_clean'])
    
    # Если метки есть - обучаем
    if label_column:
        y = df[label_column]
        
        # Разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Обучение
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Оценка
        predictions = model.predict(X_test)
        print(f"Точность: {accuracy_score(y_test, predictions):.3f}")
        
        # Предсказание для ВСЕХ данных
        all_predictions = model.predict(X)
        answer = (all_predictions == 1).sum()
        
        # Показать важные слова
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        
        top_class1 = pd.DataFrame({
            'word': feature_names,
            'coef': coefficients
        }).sort_values('coef', ascending=False).head(10)
        
        print("\nТоп-10 слов класса 1:")
        print(top_class1)
        
        return answer
    
    # Если меток нет - возвращаем векторы
    return X, vectorizer


# ═══════════════════════════════════════════════════════════════════════════
# СЦЕНАРИЙ 2: КЛАСТЕРИЗАЦИЯ ТЕКСТОВ
# ═══════════════════════════════════════════════════════════════════════════
def nlp_clustering(csv_file, text_column, n_clusters=2):
    """
    Кластеризация текстов
    
    Возвращает: размеры кластеров, ключевые слова
    """
    # Загрузка
    df = pd.read_csv(csv_file)
    df['text_clean'] = df[text_column].apply(preprocess_text)
    
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['text_clean'])
    
    # Кластеризация
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    df['cluster'] = clusters
    
    # Анализ кластеров
    feature_names = vectorizer.get_feature_names_out()
    
    for i in range(n_clusters):
        print(f"\n{'='*50}")
        print(f"КЛАСТЕР {i} (размер: {(clusters == i).sum()})")
        print(f"{'='*50}")
        
        # Топ-15 ключевых слов
        cluster_center = kmeans.cluster_centers_[i]
        top_indices = cluster_center.argsort()[-15:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        
        print(f"Ключевые слова: {', '.join(top_words)}")
    
    # ОТВЕТ: определить вручную, какой кластер - отзывы
    # Например, если в кластере 0 слова "отель", "номер" → это отзывы
    review_cluster = 0  # ИЗМЕНИТЬ ВРУЧНУЮ!
    
    answer = (clusters == review_cluster).sum()
    print(f"\n🎯 ОТВЕТ: {answer}")
    
    return answer


# ═══════════════════════════════════════════════════════════════════════════
# СЦЕНАРИЙ 3: ПОИСК КЛЮЧЕВЫХ СЛОВ ДОКУМЕНТА
# ═══════════════════════════════════════════════════════════════════════════
def nlp_keywords(csv_file, text_column, doc_id=0, top_n=10):
    """
    Извлечение ключевых слов для документа
    
    Параметры:
    - doc_id: индекс документа (или номер строки)
    - top_n: сколько ключевых слов вернуть
    """
    df = pd.read_csv(csv_file)
    df['text_clean'] = df[text_column].apply(preprocess_text)
    
    # Векторизация
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text_clean'])
    
    # Получить TF-IDF для нужного документа
    doc_vector = X[doc_id].toarray().flatten()
    feature_names = vectorizer.get_feature_names_out()
    
    # Топ-N слов
    keywords_df = pd.DataFrame({
        'word': feature_names,
        'tfidf': doc_vector
    }).sort_values('tfidf', ascending=False)
    
    top_keywords = keywords_df[keywords_df['tfidf'] > 0].head(top_n)
    print(top_keywords)
    
    # ОТВЕТ (если нужны слова через запятую)
    answer = ', '.join(top_keywords['word'].tolist())
    return answer


# ═══════════════════════════════════════════════════════════════════════════
# СЦЕНАРИЙ 4: ПОИСК ПОХОЖИХ ДОКУМЕНТОВ
# ═══════════════════════════════════════════════════════════════════════════
def nlp_similar_docs(csv_file, text_column, target_id, top_n=5):
    """
    Найти N самых похожих документов
    """
    df = pd.read_csv(csv_file)
    df['text_clean'] = df[text_column].apply(preprocess_text)
    
    # Векторизация
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(df['text_clean'])
    
    # Вычислить сходство
    target_vector = X[target_id]
    similarities = cosine_similarity(target_vector, X).flatten()
    
    # Топ-N (исключая сам документ)
    top_indices = similarities.argsort()[-top_n-1:][::-1]
    top_indices = [idx for idx in top_indices if idx != target_id][:top_n]
    
    # Если есть колонка 'id'
    if 'id' in df.columns:
        similar_ids = df.iloc[top_indices]['id'].tolist()
        answer = ', '.join(map(str, similar_ids))
    else:
        answer = ', '.join(map(str, top_indices))
    
    print(f"Похожие документы: {answer}")
    return answer


# ═══════════════════════════════════════════════════════════════════════════
# 3. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
# ═══════════════════════════════════════════════════════════════════════════

def logistic_regression_template(csv_file, target_column):
    """
    Шаблон для бинарной классификации
    
    Возвращает: количество объектов класса 1
    """
    # Загрузка
    df = pd.read_csv(csv_file)
    
    # Разделение
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Проверка баланса классов
    print("Распределение классов:")
    print(y.value_counts(normalize=True))
    
    # Обработка категориальных признаков (если есть)
    # X = pd.get_dummies(X, drop_first=True)
    
    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение
    model = LogisticRegression(
        max_iter=10000,
        class_weight='balanced',  # Для несбалансированных данных
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Оценка
    predictions = model.predict(X_test_scaled)
    print(f"\nТочность: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions))
    
    # Предсказание для ВСЕХ данных
    all_data_scaled = scaler.transform(X)
    all_predictions = model.predict(all_data_scaled)
    
    answer = (all_predictions == 1).sum()
    print(f"\n🎯 ОТВЕТ: {answer}")
    
    return answer


# ═══════════════════════════════════════════════════════════════════════════
# 4. ПАРАМЕТРИЧЕСКОЕ МОДЕЛИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════════════

# ═══ МЕТОД 1: MINIMIZE (УНИВЕРСАЛЬНЫЙ) ═══
def fit_with_minimize(x, y, model_func, init_guess):
    """
    Подбор параметров любой функции
    
    Параметры:
    - x, y: данные
    - model_func: функция модели (x, *params)
    - init_guess: начальное приближение [a, b, ...]
    """
    def error_func(params):
        predictions = model_func(x, *params)
        return np.sum((y - predictions)**2)
    
    result = minimize(error_func, x0=init_guess, method='BFGS')
    
    return result.x, result.fun


# ═══ МЕТОД 2: CURVE_FIT (ДЛЯ ЯВНЫХ ФУНКЦИЙ) ═══
def fit_with_curve_fit(x, y, model_func, init_guess):
    """
    Быстрый подбор для явных функций y = f(x)
    """
    params, _ = curve_fit(model_func, x, y, p0=init_guess, maxfev=10000)
    
    predictions = model_func(x, *params)
    error = np.sum((y - predictions)**2)
    
    return params, error


# ═══ ПРИМЕРЫ МОДЕЛЕЙ ═══

# Линейная: y = a*x + b
def linear_model(x, a, b):
    return a*x + b

# Синусоида: y = sin(a*x) + b
def sin_model(x, a, b):
    return np.sin(a*x) + b

# Полиномиальная + синус (ФИНАЛ 2024-25!)
def forecast_model(t, a1, a2, a3, a4):
    """model(t) = a1 + a2*t + a3*t² + a4*sin²(2π*t/13)"""
    return a1 + a2*t + a3*t**2 + a4*np.sin(2*np.pi*t/13)

# Эллипс (НЕЯВНОЕ УРАВНЕНИЕ! - ФИНАЛ 2023-24)
def fit_ellipse(x, y):
    """x²/a² + y²/b² = 1"""
    def ellipse_error(params):
        a, b = params
        residuals = (x**2 / a**2) + (y**2 / b**2) - 1
        return np.sum(residuals**2)
    
    # Начальное приближение
    init_guess = [np.abs(x).max() * 1.2, np.abs(y).max() * 1.2]
    
    result = minimize(ellipse_error, x0=init_guess, method='BFGS')
    a, b = result.x
    
    return a, b, result.fun


# ═══ ПРИМЕР: ПРОГНОЗИРОВАНИЕ (ФИНАЛ 2024-25) ═══
def forecast_example():
    """
    Прогноз продаж на 12 недель
    """
    # Загрузка данных (104 недели)
    df = pd.read_csv(r"forecast1.csv")
    # df = df.dropna()
    df.loc[52, "Продажи"] = (df.iloc[51]["Продажи"]+df.iloc[53]["Продажи"])/2
    t = np.arange(len(df))
    sales = df['Продажи'].values
    
    # Подбор параметров
    params, error = fit_with_curve_fit(
        t, sales, forecast_model, 
        init_guess=[sales.mean(), 0, 0, 0]
    )
    
    print(f"Параметры: {params}")
    print(f"Ошибка: {error:.2f}")
    
    # Прогноз на 12 недель
    future_t = np.arange(104, 104+12)
    forecast = forecast_model(future_t, *params)
    
    answer = ', '.join([f"{x:.2f}" for x in forecast])
    print(f"\n🎯 ОТВЕТ: {answer}")
    
    true = np.array([653.07, 675.12, 694.82, 710.26, 720.55, 726.0, 728.04, 728.92, 731.16, 737.01, 747.9, 764.13])
    pred = np.array([float(f"{pred:.2f}") for pred in forecast])

    diff = true - pred
    norm = np.linalg.norm(diff)
    score = round(max(20 - norm / 6, 0))

    print("Скор:", score)

    return answer, score

def grid_search(x, y, model_func, param_ranges):
    """
    param_ranges: словарь {'param_name': (min, max, steps)}
    Пример: {'a': (1, 3, 100), 'b': (0, 1, 100)}
    """
    from itertools import product
    from tqdm import tqdm
    
    # Создать сетку параметров
    grids = [np.linspace(min_val, max_val, steps) 
             for min_val, max_val, steps in param_ranges.values()]
    
    best_error = float('inf')
    best_params = None
    
    for params in tqdm(product(*grids), total=np.prod([steps for _, _, steps in param_ranges.values()])):
        predictions = model_func(x, *params)
        error = np.sum((y - predictions)**2)
        
        if error < best_error:
            best_error = error
            best_params = params
    
    return best_params, best_error

# ═══════════════════════════════════════════════════════════════════════════
# 5. PANDAS ШПАРГАЛКА (для тестовых заданий)
# ═══════════════════════════════════════════════════════════════════════════

def pandas_cheatsheet():
    """
    Типовые операции для тестовых заданий
    """
    df = pd.read_csv('data.csv')
    
    # ═══ БАЗОВЫЕ ОПЕРАЦИИ ═══
    
    # Количество строк
    answer = len(df)
    
    # Количество с условием
    answer = len(df[df['price'] < 1000])
    
    # Два условия (И)
    answer = len(df[(df['area'] >= 70) & (df['metro_km'] < 1)])
    
    # Два условия (ИЛИ)
    answer = len(df[(df['floor'] == 1) | (df['floor'] == 10)])
    
    # Среднее значение
    answer = df['price'].mean()
    
    # Количество меньше среднего
    answer = len(df[df['price'] < df['price'].mean()])
    
    # Максимум/минимум
    answer = df['price'].max()
    answer = df['price'].min()
    
    # Сортировка и выбор
    answer = df.sort_values('price').iloc[0]['price']  # Минимальная цена
    
    # ═══ ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ═══
    
    # Колонки с пропусками
    cols_with_na = df.columns[df.isnull().any()].tolist()
    
    df[df.isnull().any(axis=1)]
    
    # Заполнить пропуски средним
    df['column'].fillna(df['column'].mean(), inplace=True)
    
    # Заполнить линейной интерполяцией (ВАЖНО ДЛЯ ВРЕМЕННЫХ РЯДОВ!)
    df['temperature'].interpolate(method='linear', inplace=True)
    
    # ═══ КОРРЕЛЯЦИЯ ═══
    
    # Корреляция с целевой переменной
    correlations = df.corr()['price'].abs()
    
    # Три наименьших (исключая саму цену)
    answer = correlations.nsmallest(4)[1:4].index.tolist()
    
    # Три наибольших
    answer = correlations.nlargest(4)[1:4].index.tolist()
    
    # ═══ ГРУППИРОВКА ═══
    
    # Среднее по группам
    answer = df.groupby('city')['price'].mean()
    
    # Количество по группам
    answer = df.groupby('category').size()
    
    # ═══ СЛОЖНЫЕ ЗАПРОСЫ ═══
    
    # Самая дешевая квартира с условиями
    filtered = df[(df['area'] >= 90) & (df['kad_km'] < 1)]
    answer = filtered['price'].min()
    
    # Район с максимальным числом объектов
    district_counts = df['district'].value_counts()
    richest_district = district_counts.idxmax()
    answer = df[df['district'] == richest_district]['price'].min()
    
    # Среднее с условием
    answer = df[df['center_km'] < 3]['price_per_sqm'].mean()
    
    return answer


# ═══════════════════════════════════════════════════════════════════════════
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    
    # ═══ NLP: КЛАССИФИКАЦИЯ ═══
    # answer = nlp_classification('texts.csv', 'text', 'label')
    
    # ═══ NLP: КЛАСТЕРИЗАЦИЯ ═══
    # answer = nlp_clustering('texts.csv', 'text', n_clusters=2)
    
    # ═══ NLP: КЛЮЧЕВЫЕ СЛОВА ═══
    # answer = nlp_keywords('docs.csv', 'text', doc_id=42, top_n=10)
    
    # ═══ ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ═══
    # answer = logistic_regression_template('data.csv', 'target')
    
    # ═══ ПАРАМЕТРИЧЕСКОЕ МОДЕЛИРОВАНИЕ ═══
    # Пример: подбор параметров синусоиды
    # x = np.linspace(-2*np.pi, 2*np.pi, 100)
    # y = np.sin(2*x) + 0.7 + np.random.normal(0, 0.1, 100)
    # params, error = fit_with_minimize(x, y, sin_model, init_guess=[2, 0.7])
    # print(f"a={params[0]:.3f}, b={params[1]:.3f}, error={error:.3f}")
    # ═══ ЭЛЛИПС ═══
    # df = pd.read_excel(r"DataModel1.xls", header=None)
    # df.columns = ["X", "Y"]
    # x_data = df["X"].to_numpy()
    # y_data = df["Y"].to_numpy()
    # a, b, error = fit_ellipse(x_data, y_data)
    # print(f"a={a:.3f}, b={b:.3f}")
    
    pass


# ═══════════════════════════════════════════════════════════════════════════
# КРИТИЧЕСКИ ВАЖНЫЕ МОМЕНТЫ
# ═══════════════════════════════════════════════════════════════════════════
"""
1. NLTK ДАННЫЕ: Скачать!
   nltk.download('stopwords')
   nltk.download('punkt')

2. ПРОПУСКИ В ДАННЫХ: 
   - Временные ряды → interpolate(method='linear')
   - Обычные данные → fillna(mean)

3. КОРРЕЛЯЦИЯ:
   - Искать МИНИМАЛЬНУЮ по МОДУЛЮ для нерелевантных признаков

4. TF-IDF:
   - max_features=5000 - достаточно для большинства задач
   - ngram_range=(1, 2) - униграммы + биграммы

5. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ:
   - class_weight='balanced' - для несбалансированных данных
   - StandardScaler() - ОБЯЗАТЕЛЬНО нормализовать!

6. ПАРАМЕТРИЧЕСКОЕ МОДЕЛИРОВАНИЕ:
   - Если функция НЕЯВНАЯ → minimize()
   - Если функция ЯВНАЯ → curve_fit()
   - Начальное приближение важно!

7. ФОРМАТ ОТВЕТОВ:
   - Целое число → просто число
   - Список чисел → "1.23, 4.56, 7.89"
   - Список слов → "слово1, слово2, слово3"
"""
