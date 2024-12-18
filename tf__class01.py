import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('train.csv')

# Объединим имена в один столбец и создадим метки (0 для мальчиков, 1 для девочек)
data['Label'] = data['Мальчики'].notnull().astype(int)
names = data['Мальчики'].fillna('') + ' ' + data['Девочки'].fillna('')

# Используем LabelEncoder для преобразования имен в числовые значения
encoder = LabelEncoder()
encoded_names = encoder.fit_transform(names)

# Создаем входные данные и метки
X = encoded_names.reshape(-1, 1)  # Преобразуем в двумерный массив
y = data['Label'].values

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Embedding(input_dim=len(encoder.classes_), output_dim=8),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели и сохранение истории
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Оценка модели
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Построение графиков потерь и точности
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Потери')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Точность')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()

plt.show()