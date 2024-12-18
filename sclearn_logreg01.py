import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
data = pd.read_csv('train.csv')

# Предобработка данных
# Преобразуем текстовые метки в числовые значения
le = LabelEncoder()
data['Мальчики'] = le.fit_transform(data['Мальчики'])
data['Девочки'] = le.fit_transform(data['Девочки'])

# Создаем целевую переменную (0 - мальчик, 1 - девочка)
X = data[['Мальчики', 'Девочки']]
y = (data['Девочки'] > 0).astype(int)  # Если имя девочки, то 1, иначе 0

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(report)

# Визуализация результатов
plt.figure(figsize=(10, 5))

# График распределения предсказанных классов
plt.subplot(1, 2, 1)
sns.countplot(x=y_pred)
plt.title('Распределение предсказанных классов')
plt.xlabel('Класс')
plt.ylabel('Количество')
plt.xticks(ticks=[0, 1], labels=['Мальчик', 'Девочка'])

# Построение матрицы путаницы
plt.subplot(1, 2, 2)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Мальчик', 'Девочка'], yticklabels=['Мальчик', 'Девочка'])
plt.title('Confusion Matrix')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')

plt.tight_layout()
plt.show()


