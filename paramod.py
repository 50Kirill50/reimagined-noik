import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ============================================================
# МЕТОД 1: SCIPY.OPTIMIZE.MINIMIZE (УНИВЕРСАЛЬНЫЙ)
# ============================================================
# Когда использовать: для ЛЮБЫХ функций, особенно неявных
# Плюсы: работает всегда, гибкий
# Минусы: нужно писать функцию ошибки вручную

def minimize_method(x, y, model_func, init_guess):
    """
    Параметры:
    - x, y: данные
    - model_func: функция модели, принимает (x, *params)
    - init_guess: начальное приближение [a, b, ...]
    """
    
    def error_function(params, x, y):
        predictions = model_func(x, *params)
        return np.sum((y - predictions)**2)
    
    result = minimize(
        error_function,
        x0=init_guess,
        args=(x, y),
        method='BFGS'  # Градиентный спуск
    )
    
    return result.x, result.fun

# Пример 1: Линейная модель y = a*x + b
def linear_model(x, a, b):
    return a*x + b

params, error = minimize_method(x, y, linear_model, init_guess=[1, 0])
print(f"Параметры: {params}, Ошибка: {error:.3f}")

# Пример 2: Синусоида y = sin(a*x) + b
def sin_model(x, a, b):
    return np.sin(a*x) + b

params, error = minimize_method(x, y, sin_model, init_guess=[2, 0.7])

# Пример 3: НЕЯВНОЕ УРАВНЕНИЕ (эллипс)
def ellipse_error(params, x, y):
    a, b = params
    return np.sum(((x**2/a**2) + (y**2/b**2) - 1)**2)

init_guess = [x.abs().max(), y.abs().max()]
result = minimize(ellipse_error, x0=init_guess, args=(x, y), method='BFGS')
a, b = result.x

# ============================================================
# МЕТОД 2: SCIPY.OPTIMIZE.CURVE_FIT (ДЛЯ ЯВНЫХ ФУНКЦИЙ)
# ============================================================
# Когда использовать: для ЯВНЫХ функций y = f(x, params)
# Плюсы: проще синтаксис, автоматически считает ошибки
# Минусы: НЕ работает для неявных уравнений (где больше двух неизвестных)!

def curve_fit_method(x, y, model_func, init_guess):
    """
    model_func: функция вида y = f(x, a, b, ...)
    """
    params, _ = curve_fit(model_func, x, y, p0=init_guess, maxfev=10000)
    
    predictions = model_func(x, *params)
    error = np.sum((y - predictions)**2)
    
    return params, error

# Пример: Синусоида
def sin_model(x, a, b):
    return np.sin(a*x) + b

params, error = curve_fit_method(x, y, sin_model, init_guess=[2, 0.7])

# ============================================================
# МЕТОД 3: SKLEARN.LINEAR_MODEL (ТОЛЬКО ДЛЯ ЛИНЕЙНЫХ)
# ============================================================
# Когда использовать: для ЛИНЕЙНЫХ моделей вида y = a1*f1(x) + a2*f2(x) + ...
# Плюсы: очень быстрый, не нужен initial guess
# Минусы: работает ТОЛЬКО для линейных комбинаций признаков!

def sklearn_method(x, y, feature_functions):
    """
    feature_functions: список функций для признаков
    Пример: [lambda x: x, lambda x: x**2, lambda x: np.sin(x)]
    """
    # Создать матрицу признаков
    X = np.column_stack([f(x) for f in feature_functions])
    
    model = LinearRegression()
    model.fit(X, y)
    
    params = list(model.coef_) + [model.intercept_]
    predictions = model.predict(X)
    error = np.sum((y - predictions)**2)
    
    return params, error

# Пример: y = a1*x + a2*x² + a3*sin(x) + b
features = [lambda x: x, lambda x: x**2, lambda x: np.sin(x)]
params, error = sklearn_method(x, y, features)
# params = [a1, a2, a3, b]

# ============================================================
# МЕТОД 4: ПЕРЕБОР ПО СЕТКЕ (КОГДА ВСЁ ОСТАЛЬНОЕ НЕ РАБОТАЕТ)
# ============================================================
# Когда использовать: когда не можешь подобрать initial guess
# Плюсы: гарантированно найдет глобальный минимум (если сетка плотная)
# Минусы: ОЧЕНЬ медленный!

def grid_search_simple(x, y, param_range_x1, param_range_y1, param_range_x2, param_range_y2):
    # Диапазон параметров
    a_values = np.linspace(param_range_x1, param_range_y1, 400) # Значения
    # b_values = np.linspace(param_range_x2, param_range_y2, 400) # Значения
    best_error = float('inf')
    best_a = None

    for a in a_values:
        # for b in b_values:
            error = sum((y_i - a * x_i)**2 for x_i, y_i in zip(x, y))  # Функция + x + y
            if error < best_error:
                best_error = error
                best_a = a

    print(f"Лучший параметр: {best_a}")


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

# Пример
def sin_model(x, a, b):
    return np.sin(a*x) + b

param_ranges = {'a': (1.9, 2.1, 100), 'b': (0.6, 0.8, 100)}
params, error = grid_search(x, y, sin_model, param_ranges)

# ============================================================
# КОМБИНИРОВАННЫЙ ПОДХОД (РЕКОМЕНДУЕТСЯ!)
# ============================================================
# 1. Сначала грубый перебор для поиска области
# 2. Потом minimize/curve_fit для точной настройки

# Шаг 1: Грубая сетка
param_ranges = {'a': (1, 3, 10), 'b': (0, 1, 10)}
rough_params, _ = grid_search(x, y, sin_model, param_ranges)

# Шаг 2: Точная оптимизация
final_params, error = minimize_method(x, y, sin_model, init_guess=rough_params)
