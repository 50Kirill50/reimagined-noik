# ═══════════════════════════════════════════════════════════════════════════
# ВИЗУАЛИЗАЦИИ И ДИАГНОСТИКА
# ═══════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import numpy as np

# ═══ 1. ПАРАМЕТРИЧЕСКОЕ МОДЕЛИРОВАНИЕ ═══
def visualize_fit(x, y, model_func, params):
    """
    Проверить качество подгонки
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Данные и модель
    axes[0].scatter(x, y, alpha=0.5, s=20, label='Данные')
    
    x_smooth = np.linspace(x.min(), x.max(), 500)
    y_model = model_func(x_smooth, *params)
    axes[0].plot(x_smooth, y_model, 'r-', linewidth=2, label='Модель')
    
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Подгонка (параметры: {params})')
    
    # График 2: Остатки (residuals)
    residuals = y - model_func(x, *params)
    axes[1].scatter(x, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Остатки')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Остатки (должны быть случайными!)')
    
    plt.tight_layout()
    plt.show()
    
    # Статистика
    error = np.sum(residuals**2)
    print(f"Квадратичная ошибка: {error:.6f}")
    print(f"Средняя абсолютная ошибка: {np.abs(residuals).mean():.6f}")


# ═══ 2. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ ═══
def visualize_classification(X, y, model, feature_names=None):
    """
    Визуализация для классификации
    """
    from sklearn.metrics import confusion_matrix
    
    # Предсказания
    y_pred = model.predict(X)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    # График 1: Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    im = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xlabel('Предсказание')
    axes[0].set_ylabel('Истина')
    axes[0].set_title('Матрица ошибок')
    
    # Добавить числа
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, cm[i, j], ha='center', va='center', 
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
    
    # График 2: Распределение вероятностей
    proba = model.predict_proba(X)[:, 1]
    axes[1].hist(proba[y==0], bins=30, alpha=0.5, label='Класс 0')
    axes[1].hist(proba[y==1], bins=30, alpha=0.5, label='Класс 1')
    axes[1].axvline(x=0.5, color='r', linestyle='--', label='Порог')
    axes[1].set_xlabel('Вероятность класса 1')
    axes[1].set_ylabel('Частота')
    axes[1].legend()
    axes[1].set_title('Распределение вероятностей')
    
    # График 3: Важность признаков
    if feature_names is not None:
        coef = model.coef_[0]
        indices = np.argsort(np.abs(coef))[-10:]  # Топ-10
        
        axes[2].barh(range(len(indices)), coef[indices])
        axes[2].set_yticks(range(len(indices)))
        axes[2].set_yticklabels([feature_names[i] for i in indices])
        axes[2].set_xlabel('Вес')
        axes[2].set_title('Топ-10 важных признаков')
        axes[2].axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.show()


# ═══ 3. NLP - КЛАСТЕРИЗАЦИЯ ═══
def visualize_clusters(X, clusters, vectorizer, n_top_words=10):
    """
    Визуализация кластеров текстов
    """
    from sklearn.decomposition import PCA
    
    # PCA для визуализации в 2D
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X.toarray())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # График 1: Кластеры в 2D
    for i in np.unique(clusters):
        mask = clusters == i
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], 
                       label=f'Кластер {i}', alpha=0.6, s=20)
    
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend()
    axes[0].set_title('Кластеры (PCA проекция)')
    axes[0].grid(True, alpha=0.3)
    
    # График 2: Размеры кластеров
    sizes = [np.sum(clusters == i) for i in np.unique(clusters)]
    axes[1].bar(np.unique(clusters), sizes)
    axes[1].set_xlabel('Номер кластера')
    axes[1].set_ylabel('Размер')
    axes[1].set_title('Размеры кластеров')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    # Вывести ключевые слова
    print("\nКЛЮЧЕВЫЕ СЛОВА ПО КЛАСТЕРАМ:")
    print("="*60)
    
    feature_names = vectorizer.get_feature_names_out()
    
    # Вычислить средний TF-IDF для каждого кластера
    for i in np.unique(clusters):
        mask = clusters == i
        cluster_docs = X[mask]
        
        # Средний вектор кластера
        cluster_center = cluster_docs.mean(axis=0).A1
        
        # Топ-N слов
        top_indices = cluster_center.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[idx] for idx in top_indices]
        
        print(f"\nКластер {i} (размер: {np.sum(mask)}):")
        print(f"  {', '.join(top_words)}")


# ═══ 4. PANDAS - ИССЛЕДОВАНИЕ ДАННЫХ ═══
def explore_dataframe(df):
    """
    Быстрый обзор датасета
    """
    print("="*60)
    print("ОБЗОР ДАТАСЕТА")
    print("="*60)
    
    print(f"\nРазмер: {df.shape[0]} строк × {df.shape[1]} колонок")
    print(f"\nКолонки: {list(df.columns)}")
    
    print("\nТипы данных:")
    print(df.dtypes)
    
    print("\nПропущенные значения:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(missing[missing > 0])
    else:
        print("Нет пропусков!")
    
    print("\nСтатистика числовых колонок:")
    print(df.describe())
    
    # Визуализация
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    n_cols = len(numeric_cols)
    
    if n_cols > 0:
        fig, axes = plt.subplots(1, min(n_cols, 3), figsize=(15, 4))
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(numeric_cols[:3]):
            axes[i].hist(df[col].dropna(), bins=30, edgecolor='black')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Частота')
            axes[i].set_title(f'Распределение {col}')
            axes[i].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
