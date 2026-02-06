# ============ ИМПОРТЫ ============
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np

# ============ ЗАГРУЗКА ДАННЫХ ============
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

print(y.value_counts())

# ФИЧИ

def add_statistical_features(signal):
    """Добавить статистические признаки к сигналу"""
    signal = np.array(signal)
    features = [
        signal.mean(),                    # Среднее
        signal.std(),                     # Стандартное отклонение
        signal.max(),                     # Максимум
        signal.min(),                     # Минимум
        signal.max() - signal.min(),      # Размах
        np.median(signal),                # Медиана
        np.percentile(signal, 25),        # 25-й процентиль
        np.percentile(signal, 75),        # 75-й процентиль
        (signal > signal.mean()).sum(),  # Сколько выше среднего ?
        np.abs(np.diff(signal)).mean()   # Средняя скорость изменения
    ]
    return list(signal) + features

# Применить
X = np.array([add_features(row) for row in X])

# Применить к данным
X = [add_statistical_features(x) for x in X]

# ============ РАЗДЕЛЕНИЕ ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============ НОРМАЛИЗАЦИЯ ============
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============ ОБУЧЕНИЕ ============
# from sklearn.linear_model import LogisticRegressionCV

# model = LogisticRegressionCV(
#     cv=5,  # Кросс-валидация
#     max_iter=20000,
#     random_state=42
# )

# multi_class='multinomial'

model = LogisticRegression(max_iter=10000, class_weight='balanced')
model.fit(X_train_scaled, y_train)

# После обучения можно посмотреть веса
coefficients = model.coef_[0]
feature_importance = pd.DataFrame({
    'feature': range(len(coefficients)),
    'weight': coefficients
}).sort_values('weight', ascending=False)

print(feature_importance.head(10))

# ============ ПРЕДСКАЗАНИЕ ============
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)
custom_predictions = (probabilities[:, 1] > 0.7).astype(int)

print(accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# ============ ПОДСЧЕТ ОТВЕТА ============
answer = (predictions == 1).sum()
print(f"Ответ: {answer}")
