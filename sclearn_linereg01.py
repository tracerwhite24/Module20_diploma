import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

""" Переименование столбцов для удобства """
data.columns = ['Мальчики', 'Девочки', 'Год']

""" Создаем новый датафрейм с количеством упоминаний для каждого имени """
np.random.seed(0)
data['Количество'] = np.random.randint(1, 1000, size=len(data))

data['Год'] = pd.to_numeric(data['Год'], errors='coerce')

name_to_analyze = 'Александр'
name_data = data[data['Мальчики'] == name_to_analyze]

""" Проверяем, есть ли данные для данного имени """
if name_data.empty:
    print(f"Нет данных для имени {name_to_analyze}.")
else:
    X = name_data[['Год']]  # Независимая переменная (Год)
    y = name_data['Количество']  # Зависимая переменная (Количество)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    """ Создание и обучение модели линейной регрессии """
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    """ Визуализация результатов """
    plt.scatter(X_test, y_test, color='blue', label='Фактические данные')
    plt.scatter(X_test, y_pred, color='red', label='Предсказанные данные')
    plt.plot(X_test, y_pred, color='green', linewidth=2, label='Линейная регрессия')
    plt.title(f'Линейная регрессия для имени {name_to_analyze}')
    plt.xlabel('Год')
    plt.ylabel('Количество упоминаний')
    plt.legend()
    plt.show()

    """ Оценка модели """
    print(f'Коэффициенты: {model.coef_}')
    print(f'Перехват: {model.intercept_}')
