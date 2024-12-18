import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('train.csv')

data['Год'] = pd.to_numeric(data['Год'], errors='coerce')

# Удаление строк с пропусками только из столбца 'Год'
data.dropna(subset=['Год'], inplace=True)

# Подготовка данных для мальчиков
X = data['Год'].values.reshape(-1, 1)  # Признаки (Год)
y_boys = data['Мальчики'].values  # Целевая переменная (Мальчики)

# Кодирование меток для мальчиков
label_encoder_boys = LabelEncoder()
y_boys_encoded = label_encoder_boys.fit_transform(y_boys)

# Разделение данных на обучающую и тестовую выборки для мальчиков
X_train, X_test, y_train_boys, y_test_boys = train_test_split(X, y_boys_encoded, test_size=0.2, random_state=42)

# Подготовка данных для девочек
y_girls = data['Девочки'].values  # Целевая переменная (Девочки)

# Кодирование меток для девочек
label_encoder_girls = LabelEncoder()
y_girls_encoded = label_encoder_girls.fit_transform(y_girls)

# Разделение данных на обучающую и тестовую выборки для девочек
X_train_girls, X_test_girls, y_train_girls, y_test_girls = train_test_split(X, y_girls_encoded, test_size=0.2, random_state=42)

# Преобразование данных в тензоры PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_boys_tensor = torch.tensor(y_train_boys, dtype=torch.long)  # Целевые переменные как long

X_train_tensor_girls = torch.tensor(X_train_girls, dtype=torch.float32)
y_train_girls_tensor = torch.tensor(y_train_girls, dtype=torch.long)  # Целевые переменные как long

# Определение модели
class SimpleModel(nn.Module):
    def __init__(self, output_size):  # Принимаем размер выходного слоя как параметр
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 10)  # Скрытый слой
        self.output = nn.Linear(10, output_size)  # Выходной слой

    def forward(self, x):
        x = torch.relu(self.fc(x))
        return self.output(x)

# Инициализация моделей, функции потерь и оптимизаторов
model_boys = SimpleModel(len(label_encoder_boys.classes_))
criterion_boys = nn.CrossEntropyLoss()  # Для многоклассовой классификации
optimizer_boys = optim.Adam(model_boys.parameters(), lr=0.001)

model_girls = SimpleModel(len(label_encoder_girls.classes_))
criterion_girls = nn.CrossEntropyLoss()  # Для многоклассовой классификации
optimizer_girls = optim.Adam(model_girls.parameters(), lr=0.001)

# Списки для хранения потерь во время обучения
losses_boys = []
losses_girls = []

# Обучение модели для мальчиков
for epoch in range(100):  # Количество эпох
    model_boys.train()

    optimizer_boys.zero_grad()  # Обнуление градиентов

    # Прямой проход
    outputs_boys = model_boys(X_train_tensor)

    # Вычисление функции потерь
    loss_boys = criterion_boys(outputs_boys, y_train_boys_tensor)

    # Обратный проход и оптимизация
    loss_boys.backward()
    optimizer_boys.step()

    losses_boys.append(loss_boys.item())  # Сохранение потерь

    if (epoch + 1) % 10 == 0:  # Печать каждые 10 эпох
        print(f'Epoch [{epoch + 1}/100], Loss (Boys): {loss_boys.item():.4f}')

# Обучение модели для девочек
for epoch in range(100):  # Количество эпох
    model_girls.train()

    optimizer_girls.zero_grad()  # Обнуление градиентов

    # Прямой проход
    outputs_girls = model_girls(X_train_tensor_girls)

    # Вычисление функции потерь
    loss_girls = criterion_girls(outputs_girls, y_train_girls_tensor)

    # Обратный проход и оптимизация
    loss_girls.backward()
    optimizer_girls.step()

    losses_girls.append(loss_girls.item())  # Сохранение потерь

    if (epoch + 1) % 10 == 0:  # Печать каждые 10 эпох
        print(f'Epoch [{epoch + 1}/100], Loss (Girls): {loss_girls.item():.4f}')

# Визуализация потерь во время обучения
plt.figure(figsize=(12, 6))

# График потерь для мальчиков
plt.subplot(1, 2, 1)
plt.plot(losses_boys, label='Потери (мальчики)', color='blue')
plt.title('Потери во время обучения (мальчики)')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()

# График потерь для девочек
plt.subplot(1, 2, 2)
plt.plot(losses_girls, label='Потери (девочки)', color='pink')
plt.title('Потери во время обучения (девочки)')
plt.xlabel('Эпохи')
plt.ylabel('Потери')
plt.legend()

plt.tight_layout()
plt.show()

# Предсказание для мальчиков
model_boys.eval()
with torch.no_grad():
    predictions_boys = model_boys(torch.tensor(X_test, dtype=torch.float32))

# Преобразование предсказаний в классы для мальчиков
_, predicted_classes_boys = torch.max(predictions_boys, 1)

# Оценка производительности для мальчиков
accuracy_boys = (predicted_classes_boys.numpy() == y_test_boys).mean()
print(f'Точность на тестовых данных (мальчики): {accuracy_boys:.4f}')

# Предсказание для девочек
model_girls.eval()
with torch.no_grad():
    predictions_girls = model_girls(torch.tensor(X_test_girls, dtype=torch.float32))

# Преобразование предсказаний в классы для девочек
_, predicted_classes_girls = torch.max(predictions_girls, 1)

# Оценка производительности для девочек
accuracy_girls = (predicted_classes_girls.numpy() == y_test_girls).mean()
print(f'Точность на тестовых данных (девочки): {accuracy_girls:.4f}')

