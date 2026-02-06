# ═══════════════════════════════════════════════════════════════════════════
# КАК НАЙТИ НАЧАЛЬНОЕ ПРИБЛИЖЕНИЕ (init_guess)
# ═══════════════════════════════════════════════════════════════════════════

"""
ГЛАВНОЕ ПРАВИЛО: Смотри на данные и график!

Методы поиска init_guess:
"""
import numpy as np
import matplotlib.pyplot as plt


# ═══ МЕТОД 1: ВИЗУАЛИЗАЦИЯ (ВСЕГДА НАЧИНАЙ С ЭТОГО!) ═══
def find_init_guess_visual(x, y):
    """
    1. Построй график данных
    2. Прикинь параметры на глаз
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5, s=10)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.title('Данные - смотрим на графике!')
    plt.show()
    
    # ЧТО ИСКАТЬ НА ГРАФИКЕ:
    # Для y = a*x + b:
    #   a (наклон) = (y_max - y_min) / (x_max - x_min)
    #   b (сдвиг) = y при x=0 (пересечение с осью Y)
    
    # Для y = sin(a*x) + b:
    #   a (частота) = число полных колебаний / длина интервала x
    #   b (сдвиг вверх) = (y_max + y_min) / 2
    
    # Для эллипса x²/a² + y²/b² = 1:
    #   a = max(|x|) * 1.2  # С запасом!
    #   b = max(|y|) * 1.2


# ═══ ПРИМЕРЫ ДЛЯ РАЗНЫХ ФУНКЦИЙ ═══

# ЛИНЕЙНАЯ: y = a*x + b
def init_linear(x, y):
    a = (y.max() - y.min()) / (x.max() - x.min())  # Наклон
    b = y.mean() - a * x.mean()  # Сдвиг
    return [a, b]

# СИНУСОИДА: y = sin(a*x) + b
def init_sin(x, y):
    # Частота: сколько периодов на интервале?
    x_range = x.max() - x.min()
    # Прикинь на глаз или посчитай по графику
    n_periods = 2  # Примерно 2 полных колебания
    a = 2 * np.pi * n_periods / x_range
    
    # Сдвиг вверх/вниз
    b = (y.max() + y.min()) / 2
    
    return [a, b]

# КВАДРАТИЧНАЯ: y = a*x² + b*x + c
def init_quadratic(x, y):
    # Если парабола вверх → a > 0
    # Если парабола вниз → a < 0
    a = 0.1 if y[len(y)//2] > y.min() else -0.1
    b = 0
    c = y.min() if a > 0 else y.max()
    return [a, b, c]

# ЭКСПОНЕНТА: y = a*exp(b*x) + c
def init_exp(x, y):
    # Если растёт → b > 0
    # Если убывает → b < 0
    a = y[0]  # Начальное значение
    b = 0.1 if y[-1] > y[0] else -0.1
    c = 0
    return [a, b, c]

# ЭЛЛИПС: x²/a² + y²/b² = 1
def init_ellipse(x, y):
    # Взять максимумы с запасом!
    a = np.abs(x).max() * 1.2
    b = np.abs(y).max() * 1.2
    return [a, b]

# ПРОГНОЗ (ФИНАЛ 2024-25): y = a1 + a2*t + a3*t² + a4*sin²(π*t/13)
def init_forecast(t, y):
    a1 = y.mean()  # Средний уровень
    a2 = 0  # Линейный тренд (попробуй 0)
    a3 = 0  # Квадратичный тренд (попробуй 0)
    a4 = (y.max() - y.min()) / 2  # Амплитуда колебаний
    return [a1, a2, a3, a4]


# ═══ МЕТОД 2: ЕСЛИ НЕ ЗНАЕШЬ - ПЕРЕБОР ПО ГРУБОЙ СЕТКЕ ═══
def find_init_by_grid(x, y, model_func, param_ranges):
    """
    Грубый перебор для поиска области
    
    param_ranges = {
        'a': (min, max, steps),
        'b': (min, max, steps)
    }
    """
    from itertools import product
    
    grids = [np.linspace(min_v, max_v, steps) 
             for min_v, max_v, steps in param_ranges.values()]
    
    best_error = float('inf')
    best_params = None
    
    for params in product(*grids):
        predictions = model_func(x, *params)
        error = np.sum((y - predictions)**2)
        
        if error < best_error:
            best_error = error
            best_params = params
    
    return best_params

# Пример использования:
# param_ranges = {'a': (1, 3, 10), 'b': (0, 1, 10)}
# init_guess = find_init_by_grid(x, y, sin_model, param_ranges)


# ═══ МЕТОД 3: SCIPY.OPTIMIZE.DIFFERENTIAL_EVOLUTION (АВТОМАТИЧЕСКИЙ) ═══
from scipy.optimize import differential_evolution

def find_init_auto(x, y, model_func, bounds):
    """
    Автоматический поиск в заданных границах
    
    bounds = [(a_min, a_max), (b_min, b_max), ...]
    """
    def error_func(params):
        predictions = model_func(x, *params)
        return np.sum((y - predictions)**2)
    
    result = differential_evolution(error_func, bounds, seed=42)
    return result.x

# Пример:
# bounds = [(1, 3), (0, 1)]  # Для параметров a, b
# init_guess = find_init_auto(x, y, sin_model, bounds)

# Если совсем непонятно - перебор!
from scipy.optimize import brute

def error(params):
    a, b = params
    predictions = model(x, a, b)
    return np.sum((y - predictions)**2)

# Грубый перебор
ranges = [(0, 5), (0, 2)]  # Диапазоны для a и b
result = brute(error, ranges, Ns=20, finish=None)
init_guess = result  # Используй для minimize
