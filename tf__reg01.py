import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')

# Кодирование имен с использованием One-Hot Encoding
encoder_boys = OneHotEncoder(sparse_output=False)
encoder_girls = OneHotEncoder(sparse_output=False)

boys_encoded = encoder_boys.fit_transform(data[['Мальчики']])
girls_encoded = encoder_girls.fit_transform(data[['Девочки']])

# Объединение закодированных имен с исходными данными
X = data[['Год']]
y_boys = boys_encoded
y_girls = girls_encoded

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_boys_train, y_boys_test, y_girls_train, y_girls_test = train_test_split(
    X, y_boys, y_girls, test_size=0.2, random_state=42)

# Создание модели для мальчиков
model_boys = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(y_boys_train.shape[1], activation='softmax')  # Используем softmax для многоклассовой классификации
])

# Компиляция модели для мальчиков
model_boys.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели для мальчиков и сохранение истории обучения
history_boys = model_boys.fit(X_train, y_boys_train, epochs=100, verbose=1, validation_data=(X_test, y_boys_test))

# Оценка модели для мальчиков
loss_boys, accuracy_boys = model_boys.evaluate(X_test, y_boys_test)
print(f'Потери мальчиков: {loss_boys}, Точность для мальчиков: {accuracy_boys}')

# Создание модели для девочек
model_girls = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(y_girls_train.shape[1], activation='softmax')
])

# Компиляция модели для девочек
model_girls.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели для девочек и сохранение истории обучения
history_girls = model_girls.fit(X_train, y_girls_train, epochs=100, verbose=1, validation_data=(X_test, y_girls_test))

# Оценка модели для девочек
loss_girls, accuracy_girls = model_girls.evaluate(X_test, y_girls_test)
print(f'Потери для девочек: {loss_girls}, Точность для девочек: {accuracy_girls}')

# Визуализация потерь и точности для мальчиков
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_boys.history['loss'], label='train loss')
plt.plot(history_boys.history['val_loss'], label='val loss')
plt.title('Потери для модели мальчиков')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_boys.history['accuracy'], label='train accuracy')
plt.plot(history_boys.history['val_accuracy'], label='val accuracy')
plt.title('Точность модели мальчиков')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Визуализация потерь и точности для девочек
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_girls.history['loss'], label='train loss')
plt.plot(history_girls.history['val_loss'], label='val loss')
plt.title('Потери для модели девочек')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_girls.history['accuracy'], label='train accuracy')
plt.plot(history_girls.history['val_accuracy'], label='val accuracy')
plt.title('Точноть модели девочек')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Прогнозирование имен на тестовых данных
predictions_boys = model_boys.predict(X_test)
predictions_girls = model_girls.predict(X_test)

# Декодирование предсказаний
predicted_classes_boys = np.argmax(predictions_boys, axis=1)
predicted_classes_girls = np.argmax(predictions_girls, axis=1)

predicted_names_boys = encoder_boys.inverse_transform(np.eye(len(encoder_boys.categories_[0]))[predicted_classes_boys])
predicted_names_girls = encoder_girls.inverse_transform(np.eye(len(encoder_girls.categories_[0]))[predicted_classes_girls])

# Визуализация предсказанных имен
plt.figure(figsize=(12, 6))
x_axis = np.arange(len(predicted_names_boys))

plt.bar(x_axis - 0.2, predicted_names_boys.flatten(), width=0.4, label='Predicted Boys Names', color='blue')
plt.bar(x_axis + 0.2, predicted_names_girls.flatten(), width=0.4, label='Predicted Girls Names', color='pink')

plt.title('Predicted Names for Boys and Girls')
plt.xlabel('Samples')
plt.ylabel('Predicted Names')
plt.xticks(x_axis)
plt.legend()
plt.show()



