# Предиктивная аналитика больших данных

Учебный проект для демонстрации основных этапов жизненного цикла проекта предиктивной аналитики.  

## Установка 

Клонируйте репозиторий, создайте виртуальное окружение, активируйте и установите зависимости:  

```sh
git clone https://github.com/IliasAkhmetov/pabd24
cd pabd24
python -m venv venv

source venv/bin/activate  # mac or linux
.\venv\Scripts\activate   # windows

pip install -r requirements.txt
```

## Использование

### 1. Сбор данных о ценах на недвижимость 
Для парсинга информации о квартирах используйте следующий скрипт:  
``` bash 
python src/parse_cian.py 
```
Предварительно в n_rooms выбрать количество комнат, которые хотите спарсить. 

### 2. Выгрузка данных в хранилище S3 
Для доступа к хранилищу скопируйте файл .env в корень проекта.
Для выгрузки данных в хранилище S3 воспользуйтесь скриптом:

``` bash 
python src/upload_to_s3.py -i data/raw/file.csv 
```

### 3. Загрузка данных из S3 на локальную машину  
Вы можете скачать данные о квартирах, которые уже были спарсены на локальную машину:  
скрипт:

``` bash
python src/download_from_s3.py
```

### 4. Предварительная обработка данных
Скрипт для предобработки данных:  

``` bash
python src/preprocess_data.py
```

### 5. Обучение модели 
Скрипт для обучения модели

```bash 
python src/train_model.py
```
### 6. Запуск приложения flask 
Для запуска приложения flask запустите скрипт:

```bash 
python src/predict_app.py
```

### 7. Запуск приложения gunicorn
```bash 
gunicorn -b 0.0.0.0 -w 1 src.predict_app:app
```

### 8. Использование сервиса через веб интерфейс

Для использования сервиса используйте файл `web/index.html`.
'http://192.144.12.11:8000/predict'


Для запуска приложения используйте docker:
```bash 
docker run -p 8000:8000 iliasakhmetov/pabd24:latest
```
