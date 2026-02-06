import pandas as pd
# Загрузка с парсингом дат
df = pd.read_csv('data.csv', parse_dates=['date'])

# Извлечение компонентов даты
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.day_name()  # 'Monday', 'Tuesday'...
df['week'] = df['date'].dt.isocalendar().week

# Фильтрация по дате
df_2024 = df[df['year'] == 2024]
df_january = df[df['month'] == 1]

# Группировка по дате
df.groupby('month')['sales'].sum()
df.groupby('day_of_week')['purchases'].count()

target_items = set(df[df['client_id'] == 1011]['product_name'].unique())

max_overlap = 0
best_client = None

for client in df['client_id'].unique():
    if client == 1011:
        continue
    client_items = set(df[df['client_id'] == client]['product_name'].unique())
    overlap = len(target_items & client_items)
    
    if overlap > max_overlap:
        max_overlap = overlap
        best_client = client

print(f"Клиент с макс. пересечением: {best_client}")

from itertools import combinations, permutations, product

# Все пары товаров
pairs = list(combinations(df["product_name"].unique(), 2))

# Найти пару, которая встречается в наибольшем числе корзин
from collections import Counter

pair_counts = Counter()
for client in df['client_id'].unique():
    client_products = df[df['client_id'] == client]['product_name'].unique()
    for pair in combinations(client_products, 2):
        pair_counts[tuple(sorted(pair))] += 1

most_common_pair = pair_counts.most_common(1)[0]
print(f"Самая частая пара: {most_common_pair}")
