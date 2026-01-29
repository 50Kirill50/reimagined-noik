"""
Коля проводил эксперименты с некоторой электрической системой и замерял генерируемый
этой системой сигнал при различных начальных параметрах, получая массив из 61 числа. Коля
провёл 5500 экспериментов. Полученные данные представляют собой числовой массив размера
5500 на 61 и находятся в файле “WaveForm1.xlsx”
Оказалось, что изучаемая электрическая система имеет два различных режима работы.
 Режим №1 генерировал сигнал, график которого представляет собой часть синусоиды.
 Режим №2 генерировал сигнал, график которого представляет собой однократный плавно
нарастающий и плавно затухающий импульс.
По имеющемуся массиву данных помогите Коле провести классификацию экспериментов.
В качестве ответа введите количество экспериментов, в которых система работала в Режиме №2.
Ответом должно быть целое число.
Дополнительные комментарии к задаче о виде сигналов можно найти в pdf-файле
“Olymp_ClusterTask.pdf”.
"""

import numpy as np
import random

t = np.linspace(0, 6, 61)

# N1 

def n1():

    A = 0.5 + 0.1*random.random()
    w = 0.5 + random.random()
    ph = 2*np.pi * random.random()

    x = A*np.sin(t*w + ph) + A
    y = x/x.max() + 0.2 * random.random()
    return y

# N2
def n2():
    w0 = 1+4*random.random()
    gamma = random.random()

    x = t/np.sqrt((t**2 - w0**2)**2+gamma*t**2)
    y = x/x.max() + 0.2*random.random()

    return y


# x, y
n1()


X, y = [], []

for i in range(1000000):
    X.append(n1().tolist())
    y.append(0)

for i in range(1000000):
    X.append(n2().tolist())
    y.append(1)


len(X[0])

def add_features(signal):
    """Добавить статистические признаки"""
    signal = np.array(signal)
    features = [
        signal.mean(),      # Среднее
        signal.std(),       # Стандартное отклонение
        signal.max(),       # Максимум
        signal.min(),       # Минимум
        signal.max() - signal.min(),  # Размах
    ]
    return list(signal) + features

# Применить к данным
X = [add_features(signal) for signal in X]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train

# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression(max_iter=10000)

from sklearn.linear_model import LogisticRegressionCV

model = LogisticRegressionCV(
    cv=5,  # Кросс-валидация
    max_iter=20000,
    random_state=42
)

model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report

# model.predict(X_test)[:10], y_test[:10]
accuracy_score(y_test, model.predict(X_test))

print(classification_report(y_test, model.predict(X_test)))

import pandas as pd

df = pd.read_excel(r"WaveForm1.xlsx", header=None)
# interchange the index and columns axis
# df = df.T
df

ans = []
b = df.to_numpy().tolist()
b = [add_features(signal) for signal in b]
for i in b:
    i = np.array(i)
    # print(a, a.reshape(-1, 1))
    # print(scaler.transform(a.reshape(1, -1)))
    # a = scaler.transform(a.reshape(1, -1))
    # ans.append(model.predict(a).tolist()[0])

    ans.append(model.predict(scaler.transform(i.reshape(1, -1))).tolist()[0])
    # break

ans.count(1)
