# Module20_diploma
Дипломная работа по курсу Python-Разработчик

Сравнение различных библиотек для машинного обучения: scikit-learn, TensorFlow и PyTorch: Реализовать задачи классификации и регрессии с использованием scikit-learn, TensorFlow и PyTorch, сравнить их производительность и удобство использования.


Содержание: 
1.   Обзор проекта
2.   Выбор датасета для обучения модели
3.   Реализация задачи путём логистической регрессии в scikit-learn
4.   Реализация задачи путем линейной регрессии в scikit-learn
5.   Реализация задачи путём бинарной классификации в Tensorflow
6.   Реализация задачи путём регрессии в TensorFlow
7.   Реализация задачи путём логистической регрессии в PyTorch
8.   Реализация задачи путём линейной регрессии в PyTorch
9.   Заключение
10.  Приложение: список необходимых библиотек


Обзор проекта

Проект подразумевает создание шести моделей нейронных сетей, которые будут обучаться на одном датасете и выполнять задачи классификации с построением графика. Автор решил отказаться от использования готовых датасетов и создать свой самостоятельно.


Выбор датасета для обучения модели

В качестве данных, собранных в учебный файл train.csv автор подготовил статистику наиболее популярным именам для новорожденных мальчиков и девочек в Москве с 2000 по 2024 год. 
Такой датасет невелик, и в то же время содержит как строчные записи (str), так и числовые (int).
Код для создания датасета:

![image](https://github.com/user-attachments/assets/e2f8660c-d41d-4722-9eb9-aa02a8bf105d)


Реализация задачи путём логистической регрессии в scikit-learn

Автор составил код с использованием линейной модели LogisticRegression и применил визуализацию с использованием matplotlib и seaborn. 
В ходе обучения модель разделила столбцы с именами на два класса и начала обучаться определять, где мужское имя, а где женское. Как видим в результате модель определяет имена с точностью 1.0, однако, имена девочек определяются лучше. 
Вероятно, это связано с тем, что в столбце "Девочки" последние 10 записей содержат одно и то же имя.
![image](https://github.com/user-attachments/assets/2ac680fa-4830-4b17-8a23-682f67dc778b)


Реализация задачи путём линейной регрессии в scikit-learn

Автор составил код для модели линейной регрессии LinearRegression и применил визуализацию с matplotlib. В ходе обучения модель стремится установить зависимость одного из имён (Александр) и определённым годом.
Как видим на графике, модель определила, что имя Александр несколько теряет в популярности за выбранный период.Модель показывает отрицательные коэффиценты и положительный перехват.
![image](https://github.com/user-attachments/assets/bd029a83-9b11-46d1-b268-636cc373c041)


Реализация задачи путём бинарной классификации в Tensorflow

Автор использовал scikit-learn для предобработки датасета, преобразовал имена в числовые значения и создал модель с использованием tf.keras.Sequental. Модель обучается в течении 10 эпох и выводит графики точности и потерь. 
Как видим, потери снижаются, а точность растёт. Однако, модель не выводит никакой ценной информации, чему именно она научилась.
![image](https://github.com/user-attachments/assets/958bc494-84b9-4c64-8e15-9ddff94d0cb0)



Реализация задачи путём регрессии в TensorFlow

Автор создал для этой задачи две модели - одну для мальчиков и одну для девочек, которые запускаются последовательно, а затем выводят графики точности и потерь для каждой из моделей. 
Для преобразования строк в числовые величины используется One-Hot Encoding. Как видим на графике, модель обучается и точность растёт примерно до 80 эпох.
![image](https://github.com/user-attachments/assets/1882bf1d-ab48-461b-83f3-092acae4b079)



Реализация задачи путём логистической регрессии в PyTorch

Автор использовал метод Dataset, чтобы создать специальный класс, который будет работать с данными, преобразованными в числовые. Модель обучается в ходе 100 эпох а затем при помощи функции torch.no_grad старается предсказать имена
в 2025 и 2026 гг. Как видим из графика, потери то снижаются, то растут.
![image](https://github.com/user-attachments/assets/d308f2d3-4613-4661-a8d4-36755ba2eefa)


Реализация задачи путём линейной регрессии в PyTorch

Автор реализовал создание модели при помощи класса и реализовал две модели - для мальчиков и девочек - путём наследования. Модели обучаются на 100 эпохах и выводят графики потерь. 
График для мальчиков более пологий, а точность указана выше для мальчиков, чем для девочек.
![image](https://github.com/user-attachments/assets/119b1288-6b8b-4744-9377-840620f0fafc)


Заключение

Три библиотеки - sclearn, TensorFlow и PyTorch показывают разные возможности в обучении и различный функционал. Scikit-learn наиболее простая в освоении, менее требовательна к ресурсам, и её функционал ограничен.
Однако, она необходима для предварительной обработки датасета, поскольку машине для обучения необходимо преобразовать строки со словами в числительные. Также она используется для разделения данных на обучающую и тестовую выборки.
Модели TensorFlow наиболее громоздки, но и так же показывают более высокую точность.
Модели PyTorch наиболее функциональны, именно благодаря классификации с PyTorch автору удалось реализовать задачу предсказания имён в 2025 и 2026.


Приложение: список необходимых библиотек:

cloudpickle==3.1.0
contourpy==1.3.1
cycler==0.12.1
fonttools==4.55.3
gym==0.26.2
gym-notices==0.0.8
joblib==1.4.2
kiwisolver==1.4.7
matplotlib==3.9.4
numpy==2.2.0
packaging==24.2
pandas==2.2.3
pillow==11.0.0
pyparsing==3.2.0
python-dateutil==2.9.0.post0
pytz==2024.2
scikit-learn==1.6.0
scipy==1.14.1
seaborn==0.13.2
six==1.17.0
threadpoolctl==3.5.0
tzdata==2024.2
absl-py==2.1.0
astunparse==1.6.3
certifi==2024.8.30
charset-normalizer==3.4.0
cloudpickle==3.1.0
contourpy==1.3.1
cycler==0.12.1
flatbuffers==24.3.25
fonttools==4.55.3
gast==0.6.0
google-pasta==0.2.0
grpcio==1.68.1
gym==0.26.2
gym-notices==0.0.8
h5py==3.12.1
idna==3.10
joblib==1.4.2
keras==3.7.0
kiwisolver==1.4.7
libclang==18.1.1
Markdown==3.7
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
ml-dtypes==0.4.1
namex==0.0.8
opt_einsum==3.4.0
optree==0.13.1
packaging==24.2
pillow==11.0.0
protobuf==4.25.5
Pygments==2.18.0
pyparsing==3.2.0
python-dateutil==2.9.0.post0
pytz==2024.2
requests==2.32.3
rich==13.9.4
scipy==1.14.1
setuptools==75.6.0
six==1.17.0
tensorboard==2.17.1
tensorboard-data-server==0.7.2
tensorflow==2.17.0
tensorflow-intel==2.17.0
termcolor==2.5.0
threadpoolctl==3.5.0
typing_extensions==4.12.2
urllib3==2.2.3
Werkzeug==3.1.3
wheel==0.45.1
wrapt==1.17.0
contourpy==1.3.1
cycler==0.12.1
filelock==3.16.1
fonttools==4.55.3
fsspec==2024.10.0
Jinja2==3.1.4
joblib==1.4.2
kiwisolver==1.4.7
MarkupSafe==3.0.2
mpmath==1.3.0
networkx==3.4.2
packaging==24.2
pillow==11.0.0
pyparsing==3.2.0
python-dateutil==2.9.0.post0
pytz==2024.2
scipy==1.14.1
setuptools==75.6.0
six==1.17.0
sympy==1.13.1
threadpoolctl==3.5.0
torch==2.5.1
torchaudio==2.5.1
torchvision==0.20.1
typing_extensions==4.12.2

