"""
РАЗРЕШЕННЫЕ САЙТЫ НА ОЛИМПИАДЕ:
• https://docs.python.org
• https://scikit-learn.ru/
• https://pandas.pydata.org
• https://numpy.org/doc/
• https://www.nltk.org/
• https://www.geeksforgeeks.org
• https://education.yandex.ru/handbook
• https://jupyter.org/
• GitHub (публичные репозитории, БЕЗ авторизации)
• arXiv / Papers with Code / Hugging Face (открытые страницы)
• Google Colab / Kaggle (просмотр ноутбуков БЕЗ запуска)

СОДЕРЖАНИЕ:
1. ПОДГОТОВКА (NLTK, импорты)
2. ПАРАМЕТРИЧЕСКОЕ МОДЕЛИРОВАНИЕ (КАК НАЙТИ INIT_GUESS!)
3. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (расширенная версия)
4. NLP + TF-IDF (5 методов, включая правила)
5. PANDAS

═══════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# 1. ПОДГОТОВКА
# ═══════════════════════════════════════════════════════════════════════════

import pandas as pd
import numpy as np
import re
import string
from scipy.optimize import minimize, curve_fit
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# ═══ СКАЧАТЬ NLTK ДАННЫЕ (ОДИН РАЗ ПЕРЕД ОЛИМПИАДОЙ!) ═══
# import nltk
# nltk.download('stopwords')
# nltk.download('punkt')


# ═══════════════════════════════════════════════════════════════════════════
# 2. ПАРАМЕТРИЧЕСКОЕ МОДЕЛИРОВАНИЕ
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════
КРИТИЧЕСКИ ВАЖНО: КАК НАЙТИ INIT_GUESS (начальное приближение)
═══════════════════════════════════════════════════════════════════════════

МЕТОД 1: ВИЗУАЛИЗАЦИЯ (ГЛАВНЫЙ СПОСОБ!)
────────────────────────────────────────
1. Построй график данных
2. Посмотри на форму кривой
3. Оцени параметры "на глаз"

МЕТОД 2: АНАЛИЗ ДАННЫХ
────────────────────────────────────────
- Для линейной y = a*x + b:
  a ≈ (y.max() - y.min()) / (x.max() - x.min())  # Наклон
  b ≈ y.mean() - a * x.mean()                     # Смещение

- Для синусоиды y = A*sin(ω*x) + B:
  A ≈ (y.max() - y.min()) / 2      # Амплитуда
  B ≈ y.mean()                      # Смещение
  ω ≈ 2π / период                   # Частота (посмотри на график!)

- Для эллипса x²/a² + y²/b² = 1:
  a ≈ x.abs().max() * 1.2
  b ≈ y.abs().max() * 1.2

МЕТОД 3: ГРУБЫЙ ПЕРЕБОР → ТОЧНАЯ НАСТРОЙКА
────────────────────────────────────────
1. Сначала grid search на грубой сетке
2. Потом minimize/curve_fit для точности

═══════════════════════════════════════════════════════════════════════════
"""


# ═══ ФУНКЦИЯ ДЛЯ ВИЗУАЛИЗАЦИИ ДАННЫХ ═══
def visualize_data(x, y, title="График данных"):
    """
    Построить график данных - ПЕРВЫЙ ШАГ!
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, s=20, label='Данные')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    # Подсказки для init_guess
    print("═══ ПОДСКАЗКИ ДЛЯ INIT_GUESS ═══")
    print(f"x: min={x.min():.2f}, max={x.max():.2f}, mean={x.mean():.2f}")
    print(f"y: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
    print(f"Размах y: {y.max() - y.min():.2f}")
    print(f"Размах x: {x.max() - x.min():.2f}")


# ═══ ВИЗУАЛИЗАЦИЯ С ПОДОБРАННОЙ МОДЕЛЬЮ ═══
def visualize_fit(x, y, model_func, params, title="Подгонка модели"):
    """
    Показать исходные данные и подобранную модель
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, s=20, label='Данные')
    
    # Построить модель
    x_smooth = np.linspace(x.min(), x.max(), 300)
    y_model = model_func(x_smooth, *params)
    plt.plot(x_smooth, y_model, 'r-', linewidth=2, label='Модель')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


# ═══ МЕТОД 1: MINIMIZE (УНИВЕРСАЛЬНЫЙ) ═══
def fit_with_minimize(x, y, model_func, init_guess):
    """
    Подбор параметров любой функции
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

def linear_model(x, a, b):
    """y = a*x + b"""
    return a*x + b

def sin_model(x, a, b):
    """y = sin(a*x) + b"""
    return np.sin(a*x) + b

def forecast_model(t, a1, a2, a3, a4):
    """
    ФИНАЛ 2024-25: Прогноз продаж
    model(t) = a1 + a2*t + a3*t² + a4*sin²(2π*t/13)
    """
    return a1 + a2*t + a3*t**2 + a4*np.sin(2*np.pi*t/13)**2

# df = pd.read_csv(r"forecast1.csv", parse_dates=['Дата'])
# df.loc[52, "Продажи"] = (df.iloc[51]["Продажи"]+df.iloc[53]["Продажи"])/2
# df.columns = ["X", "Y"]
# t = np.arange(len(df))
# params = [16.70, 0.96, 0.0496, 23.08]
# visualize_fit(t, df["X"], forecast_model, params)

def ellipse_model(x, y):
    """
    ФИНАЛ 2023-24: Эллипс (НЕЯВНОЕ УРАВНЕНИЕ!)
    x²/a² + y²/b² = 1
    
    КАК НАЙТИ INIT_GUESS:
    1. Построить scatter plot
    2. a ≈ max(|x|) * 1.2
    3. b ≈ max(|y|) * 1.2
    """
    def error_func(params):
        a, b = params
        residuals = (x**2 / a**2) + (y**2 / b**2) - 1
        return np.sum(residuals**2)
    
    # Init guess
    init_guess = [np.abs(x).max() * 1.2, np.abs(y).max() * 1.2]
    
    result = minimize(error_func, x0=init_guess, method='BFGS')
    a, b = result.x
    
    return a, b, result.fun


# ═══ ПРИМЕР: ПОШАГОВОЕ РЕШЕНИЕ ═══
def modeling_step_by_step():
    """
    ПОЛНЫЙ ПРИМЕР: как решать задачу моделирования
    """
    # Загрузка данных
    df = pd.read_csv('data.csv')
    x = df['x'].values
    y = df['y'].values
    
    # ШАГ 1: ВИЗУАЛИЗАЦИЯ (обязательно!)
    visualize_data(x, y, "Исходные данные")
    
    # ШАГ 2: ВЫБРАТЬ МОДЕЛЬ (по графику)
    # Если похоже на синусоиду → sin_model
    # Если похоже на прямую → linear_model
    # Если похоже на параболу + синус → forecast_model
    
    # ШАГ 3: ОЦЕНИТЬ INIT_GUESS
    # Для синусоиды:
    amplitude = (y.max() - y.min()) / 2
    offset = y.mean()
    # Частота - посмотри на график, сколько периодов на интервале
    # Если 2 полных периода на [0, 2π] → частота ≈ 2
    frequency = 2.0  # Оценить по графику!
    
    init_guess = [frequency, offset]
    
    # ШАГ 4: ПОДБОР ПАРАМЕТРОВ
    params, error = fit_with_curve_fit(x, y, sin_model, init_guess)
    
    print(f"Параметры: a={params[0]:.3f}, b={params[1]:.3f}")
    print(f"Ошибка: {error:.3f}")
    
    # ШАГ 5: ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТА
    visualize_fit(x, y, sin_model, params, "Подогнанная модель")
    
    return params


# ═══════════════════════════════════════════════════════════════════════════
# 3. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (РАСШИРЕННАЯ)
# ═══════════════════════════════════════════════════════════════════════════

"""
═══════════════════════════════════════════════════════════════════════════
МЕТОДЫ КЛАССИФИКАЦИИ (когда что использовать)
═══════════════════════════════════════════════════════════════════════════

МЕТОД 1: ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (по умолчанию!)
────────────────────────────────────────
✅ Быстрая, интерпретируемая
✅ Хорошо работает с TF-IDF векторами
✅ Можно посмотреть важность признаков
❌ Только линейная граница

МЕТОД 2: NAIVE BAYES (для текстов!)
────────────────────────────────────────
✅ ОЧЕНЬ БЫСТРЫЙ
✅ Хорошо работает с текстами
✅ Не боится больших размерностей
❌ Предполагает независимость признаков

МЕТОД 3: LINEAR SVM (если LogReg не работает)
────────────────────────────────────────
✅ Хорошо разделяет классы
✅ Работает в высоких размерностях
❌ Медленнее LogReg

МЕТОД 4: LOGISTIC REGRESSION CV (автоподбор параметров)
────────────────────────────────────────
✅ Автоматически подбирает регуляризацию
✅ Надежнее базовой LogReg
❌ Чуть медленнее

═══════════════════════════════════════════════════════════════════════════
"""

def classification_comparison(X_train, X_test, y_train, y_test):
    """
    Сравнение разных методов классификации
    """
    models = {
        'LogReg': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'LogRegCV': LogisticRegressionCV(cv=5, max_iter=1000, class_weight='balanced'),
        'NaiveBayes': MultinomialNB(),
        'LinearSVM': LinearSVC(max_iter=1000, class_weight='balanced')
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = accuracy
        print(f"{name}: {accuracy:.3f}")
    
    # Выбрать лучший
    best_model_name = max(results, key=results.get)
    best_model = models[best_model_name]
    
    print(f"\nЛучшая модель: {best_model_name}")
    return best_model


def logistic_regression_full(csv_file, target_column):
    """
    ПОЛНЫЙ ШАБЛОН для бинарной классификации
    """
    # Загрузка
    df = pd.read_csv(csv_file)
    
    # Проверка баланса
    print("Распределение классов:")
    print(df[target_column].value_counts(normalize=True))
    
    # Разделение
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Обработка категориальных (если есть)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # Разделение
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение (с автоподбором параметров!)
    model = LogisticRegressionCV(
        cv=5,
        max_iter=10000,
        class_weight='balanced',
        random_state=42,
        scoring='accuracy'
    )
    model.fit(X_train_scaled, y_train)
    
    print(f"Лучший параметр C: {model.C_}")
    
    # Оценка
    predictions = model.predict(X_test_scaled)
    print(f"\nТочность: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions))
    
    # Визуализация матрицы ошибок
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.xlabel('Предсказано')
    plt.ylabel('Истина')
    plt.title('Матрица ошибок')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.show()
    
    # Важность признаков
    if hasattr(model, 'coef_'):
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(model.coef_[0])
        }).sort_values('importance', ascending=False)
        
        print("\nТоп-10 важных признаков:")
        print(feature_importance.head(10))
        
        # Визуализация
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance.head(10)['feature'], 
                 feature_importance.head(10)['importance'])
        plt.xlabel('Важность')
        plt.title('Топ-10 признаков')
        plt.tight_layout()
        plt.show()
    
    # ОТВЕТ для всех данных
    all_data_scaled = scaler.transform(X)
    all_predictions = model.predict(all_data_scaled)
    answer = (all_predictions == 1).sum()
    
    print(f"\n🎯 ОТВЕТ: {answer}")
    return answer


# ═══════════════════════════════════════════════════════════════════════════
# 4. NLP + TF-IDF
# ═══════════════════════════════════════════════════════════════════════════

# ═══ ПРЕДОБРАБОТКА ═══
def preprocess_text(text, language='russian', 
                    remove_numbers=True, 
                    remove_stopwords=True, 
                    stem=True):
    """Полная предобработка текста"""
    stop_words = set(stopwords.words(language))
    stemmer = SnowballStemmer(language)
    
    text = text.lower()
    
    if remove_numbers:
        text = re.sub(r'\d+', '', text)
    
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    
    if remove_stopwords:
        words = [w for w in words if w not in stop_words]
    
    if stem:
        words = [stemmer.stem(w) for w in words]
    
    return " ".join(words)


"""
═══════════════════════════════════════════════════════════════════════════
5 МЕТОДОВ РЕШЕНИЯ NLP ЗАДАЧ
═══════════════════════════════════════════════════════════════════════════

МЕТОД 1: TF-IDF + ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ (стандарт!)
────────────────────────────────────────
✅ Универсальный
✅ Хорошо работает в большинстве случаев

МЕТОД 2: TF-IDF + КЛАСТЕРИЗАЦИЯ (если меток нет)
────────────────────────────────────────
✅ Для неразмеченных данных
❌ Нужно вручную определить, что есть что

МЕТОД 3: ПРАВИЛА НА ОСНОВЕ КЛЮЧЕВЫХ СЛОВ (простой!)
────────────────────────────────────────
✅ Очень быстрый
✅ Интерпретируемый
❌ Нужно знать ключевые слова

МЕТОД 4: COUNT VECTORIZER + NAIVE BAYES (для текстов)
────────────────────────────────────────
✅ Быстрее TF-IDF
✅ Хорошо для коротких текстов

МЕТОД 5: ГИБРИДНЫЙ (правила + ML)
────────────────────────────────────────
✅ Сначала фильтр по правилам
✅ Потом ML для сложных случаев

═══════════════════════════════════════════════════════════════════════════
"""

# ═══ МЕТОД 3: КЛАССИФИКАЦИЯ ПО ПРАВИЛАМ ═══
def classify_by_rules(df, text_column):
    """
    Классификация на основе ключевых слов
    
    Пример: отзывы vs новости
    """
    # Предобработка
    df['clean'] = df[text_column].apply(preprocess_text)
    
    # Ключевые слова для отзывов
    review_keywords = ['отел', 'гостиниц', 'администратор', 'отдых', 
                       'номер', 'персонал', 'завтрак', 'сервис']
    
    # Ключевые слова для новостей
    news_keywords = ['новост', 'прессслужб', 'президент', 'правительств']
    
    # Классификация
    def classify(text):
        # Проверка на отзыв
        has_review_words = any(word in text for word in review_keywords)
        has_news_words = any(word in text for word in news_keywords)
        
        if has_review_words and not has_news_words:
            return 1  # Отзыв
        elif has_news_words and not has_review_words:
            return 0  # Новость
        else:
            return -1  # Непонятно
    
    df['prediction'] = df['clean'].apply(classify)
    
    # АЛЬТЕРНАТИВНЫЙ СПОСОБ (через pandas):
    mask_review = (
        (df['clean'].str.contains('отел')) | 
        (df['clean'].str.contains('гостиниц')) |
        (df['clean'].str.contains('администратор')) |
        (df['clean'].str.contains('отдых'))
    ) & (
        ~df['clean'].str.contains('новост') &
        ~df['clean'].str.contains('прессслужб')
    )
    
    df['prediction_alt'] = 0
    df.loc[mask_review, 'prediction_alt'] = 1
    
    # Статистика
    print("Распределение по правилам:")
    print(df['prediction'].value_counts())
    
    # ОТВЕТ
    answer = (df['prediction'] == 1).sum()
    print(f"\n🎯 ОТВЕТ: {answer}")
    
    return answer


# ═══ ВИЗУАЛИЗАЦИЯ ДЛЯ NLP ═══
def visualize_text_clusters(X, labels, n_samples=1000):
    """
    Визуализация кластеров текстов (PCA)
    """
    from sklearn.decomposition import PCA
    
    # Уменьшить размерность до 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X[:n_samples].toarray())
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                         c=labels[:n_samples], 
                         cmap='viridis', 
                         alpha=0.6, 
                         s=50)
    plt.colorbar(scatter)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Визуализация кластеров текстов')
    plt.grid(True, alpha=0.3)
    plt.show()


def visualize_word_importance(vectorizer, model, top_n=20):
    """
    Визуализация важности слов для классификации
    """
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Топ слов для класса 1
    top_positive = pd.DataFrame({
        'word': feature_names,
        'coef': coefficients
    }).sort_values('coef', ascending=False).head(top_n)
    
    # Топ слов для класса 0
    top_negative = pd.DataFrame({
        'word': feature_names,
        'coef': coefficients
    }).sort_values('coef', ascending=True).head(top_n)
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    axes[0].barh(top_positive['word'], top_positive['coef'])
    axes[0].set_xlabel('Коэффициент')
    axes[0].set_title('Топ-20 слов для класса 1')
    
    axes[1].barh(top_negative['word'], -top_negative['coef'])
    axes[1].set_xlabel('|Коэффициент|')
    axes[1].set_title('Топ-20 слов для класса 0')
    
    plt.tight_layout()
    plt.show()


# ═══ ПОЛНЫЙ ПРИМЕР NLP ═══
def nlp_full_pipeline(csv_file, text_column, label_column=None):
    """
    Полный пайплайн NLP с визуализацией
    """
    df = pd.read_csv(csv_file)
    
    # Предобработка
    print("Предобработка текстов...")
    df['clean'] = df[text_column].apply(preprocess_text)
    
    # Векторизация
    print("Векторизация...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['clean'])
    
    if label_column:
        # СЦЕНАРИЙ 1: КЛАССИФИКАЦИЯ
        y = df[label_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Обучение
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # Оценка
        predictions = model.predict(X_test)
        print(f"\nТочность: {accuracy_score(y_test, predictions):.3f}")
        
        # Визуализация важных слов
        visualize_word_importance(vectorizer, model)
        
        # ОТВЕТ
        all_predictions = model.predict(X)
        answer = (all_predictions == 1).sum()
        
    else:
        # СЦЕНАРИЙ 2: КЛАСТЕРИЗАЦИЯ
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Анализ кластеров
        feature_names = vectorizer.get_feature_names_out()
        for i in range(2):
            print(f"\n{'='*50}")
            print(f"КЛАСТЕР {i} (размер: {(labels == i).sum()})")
            
            cluster_center = kmeans.cluster_centers_[i]
            top_indices = cluster_center.argsort()[-15:][::-1]
            top_words = [feature_names[idx] for idx in top_indices]
            print(f"Ключевые слова: {', '.join(top_words)}")
        
        # Визуализация
        visualize_text_clusters(X, labels)
        
        # ОТВЕТ (определить вручную!)
        review_cluster = 0  # ИЗМЕНИТЬ!
        answer = (labels == review_cluster).sum()
    
    print(f"\n🎯 ОТВЕТ: {answer}")
    return answer


# ═══════════════════════════════════════════════════════════════════════════
# 5. PANDAS
# ═══════════════════════════════════════════════════════════════════════════

def pandas_cheatsheet():
    """Типовые операции для тестовых заданий"""
    df = pd.read_csv('data.csv')
    
    # ═══ БАЗОВЫЕ ОПЕРАЦИИ ═══
    answer = len(df)
    answer = len(df[df['price'] < 1000])
    answer = len(df[(df['area'] >= 70) & (df['metro_km'] < 1)])
    answer = len(df[(df['floor'] == 1) | (df['floor'] == 10)])
    
    # Среднее
    answer = df['price'].mean()
    answer = len(df[df['price'] < df['price'].mean()])
    
    # Мин/макс
    answer = df['price'].max()
    answer = df.sort_values('price').iloc[0]['price']
    
    # ═══ ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ═══
    cols_with_na = df.columns[df.isnull().any()].tolist()
    
    # Заполнение
    df['column'].fillna(df['column'].mean(), inplace=True)
    
    # ВАЖНО: Для временных рядов!
    df['temperature'].interpolate(method='linear', inplace=True)
    
    # ═══ КОРРЕЛЯЦИЯ ═══
    correlations = df.corr()['price'].abs()
    answer = correlations.nsmallest(4)[1:4].index.tolist()
    answer = correlations.nlargest(4)[1:4].index.tolist()

    # ═══ ГРУППИРОВКА ═══
    answer = df.groupby('city')['price'].mean()
    answer = df.groupby('category').size()

    # ═══ СЛОЖНЫЕ ЗАПРОСЫ ═══
    filtered = df[(df['area'] >= 90) & (df['kad_km'] < 1)]
    answer = filtered['price'].min()

    # Район с максимальным числом школ
    richest_district = df['district'].value_counts().idxmax()
    answer = df[df['district'] == richest_district]['price'].min()

    # Среднее с условием
    answer = df[df['center_km'] < 3]['price_per_sqm'].mean()

    return answer

# ═══════════════════════════════════════════════════════════════════════════
# 6. ПОЛЕЗНЫЕ ССЫЛКИ ДЛЯ ПОИСКА ИНФОРМАЦИИ НА ОЛИМПИАДЕ
# ═══════════════════════════════════════════════════════════════════════════

"""
ЧТО ИСКАТЬ НА РАЗРЕШЕННЫХ САЙТАХ:

PANDAS (https://pandas.pydata.org)
────────────────────────────────────────
• Поиск: "pandas filter by condition"
• Поиск: "pandas groupby"
• Поиск: "pandas fillna"
• Поиск: "pandas interpolate"
SCIKIT-LEARN (https://scikit-learn.ru/)
────────────────────────────────────────
• LogisticRegression - параметры, примеры
• TfidfVectorizer - параметры
• KMeans - кластеризация
SCIPY (https://docs.scipy.org)
────────────────────────────────────────
• scipy.optimize.minimize - примеры
• scipy.optimize.curve_fit
NUMPY (https://numpy.org/doc/)
────────────────────────────────────────
• Математические функции (sin, cos, exp)
• Операции с массивами
GEEKSFORGEEKS (https://www.geeksforgeeks.org)
────────────────────────────────────────
• "logistic regression python example"
• "tfidf sklearn example"
• "scipy curve fit example"
GITHUB
────────────────────────────────────────
Искать готовые решения:
• "sklearn tfidf classification github"
• "scipy minimize ellipse fitting github"
ARXIV (если нужны формулы!)
────────────────────────────────────────
• Формулы для VPD (vapor pressure deficit)
• Уравнение ван дер Ваальса
• Математические модели

СТРАТЕГИЯ ПОИСКА:

Сначала ищи в pandas/sklearn документации
Если не нашел → GeeksForGeeks
Если совсем не понятно → GitHub примеры
Формулы из условия → arXiv
"""

# ═══════════════════════════════════════════════════════════════════════════
# ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "main":
    # # ═══ ПАРАМЕТРИЧЕСКОЕ МОДЕЛИРОВАНИЕ ═══
    # modeling_step_by_step()

    # # ═══ ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ═══
    # answer = logistic_regression_full('data.csv', 'target')

    # # ═══ NLP: ПОЛНЫЙ ПАЙПЛАЙН ═══
    # answer = nlp_full_pipeline('texts.csv', 'text', 'label')

    # # ═══ NLP: ПО ПРАВИЛАМ ═══
    # answer = classify_by_rules(df, 'text')

    pass

# ═══════════════════════════════════════════════════════════════════════════
# КРИТИЧЕСКИ ВАЖНЫЕ МОМЕНТЫ
# ═══════════════════════════════════════════════════════════════════════════

"""
ВИЗУАЛИЗАЦИЯ - ВСЕГДА ПЕРВЫЙ ШАГ!
• Построить график данных
• Оценить параметры "на глаз"
• Проверить результат
INIT_GUESS:
• Для синуса: посмотри на период, амплитуду
• Для линейной: наклон ≈ (y_max - y_min) / (x_max - x_min)
• Для эллипса: a, b ≈ max|x|, max|y| * 1.2
ПРОПУСКИ В ДАННЫХ:
• Временные ряды → df['col'].interpolate(method='linear')
• Обычные данные → df['col'].fillna(mean)
КОРРЕЛЯЦИЯ:
• Искать МИНИМАЛЬНУЮ по МОДУЛЮ для нерелевантных
NLP:
• Если метки есть → TF-IDF + LogReg
• Если меток нет → Кластеризация
• Если знаешь ключевые слова → Правила
ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ:
• class_weight='balanced' - для несбалансированных
• StandardScaler() - ОБЯЗАТЕЛЬНО нормализовать!
• LogisticRegressionCV - автоподбор параметров
РАЗРЕШЕННЫЕ САЙТЫ:
• pandas.pydata.org - для работы с данными
• scikit-learn.ru - для ML
• geeksforgeeks.org - примеры кода
• GitHub - готовые решения
"""
