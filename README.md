# Gosznak ML Taks

## Задание 1: Структуры и алгоритмы
См. файл `Task_1.ipynb`.

## Задание 2 ML\DL

### Организация кода
    ├── README.md             <- Инструкция.
    │
    ├── configs    <- Параметры для обучения
    │   ├── default.yaml         <- Параметры по умолчанию
    │   ├── training             <- Дополнительные параметры для задачи классификации/деноизинга 
    |
    ├── logs        <- TensorBoard логи и веса моделей
    ├── src        <- Код

### Системные требования
* OS: Ubuntu 16.04
* Python: 3.6+
* CUDA: 10.2

<!-- Обучение проводилось на Tesla V100 -->

### Установка
1. Скопировать репозиторий `git clone https://github.com/zakajd/audio-denoising.git && cd audio-denoising`
2. Создать новое окружение `conda create --name audio-denoising torch`
3. Выполнить `pip install -r requirements.txt` для установки зависимостей


### Тестирование лучшей модели
* [Скачать](https://drive.google.com/drive/folders/1DKdpLnkZkXiI2zdgeBGjA_nSGJnKVlJY?usp=sharing) веса моделей `classification_weights.zip`, `denoising_weights.zip` в папку проекта.
* Разархивировать и переместить их в папки с логами и параметрами моделей: `unzip classification_weights.zip && mv model.chpn logs/classification_best` `unzip denoising_weights.zip && mv model.chpn logs/denoising_best`
* Классификация `python3 test.py --help`. 
* Денойзинг `python3 test.py --config_path logs/denoising_best --file_path examples/noisy/82_121544_82-121544-0008.npy`

### Обучение модели
* Отредактируйте файл настроек `configs/default.yaml` указав путь до папки с данными.
* Классификация `CUDA_VISIBLE_DEVICES=0 python3 train.py training=classification`
* Денойзинг `CUDA_VISIBLE_DEVICES=0 python3 train.py training=denoising`

### Описание решения
<!-- Для задачи классификации была использована классическая архитектура ResNet50. Первые эксперименты показали, что из-за искусственного характера наложенного шума, уже за 1 эпоху предобученная сеть получает точность 100% и дальнейшее обучение не имеет смысла. -->

Для задачи деноизинга была использована архитектура типа Unet с предобученным энкодером SeResNet50. Обучение с MSE в качестве лосса дало сглаженный результат. Дообучение с L1 лоссом улучшило результат визуально, но сделало хуже по метрике.

### Пример 1
Шумная | Денойзинг | Чистая

![noisy](https://user-images.githubusercontent.com/15848838/97484341-ad055380-1969-11eb-97bc-eed7d5c8a42b.png)

![denoised](https://user-images.githubusercontent.com/15848838/97484420-ca3a2200-1969-11eb-8ee4-61c954d3464d.png)

![clean](https://user-images.githubusercontent.com/15848838/97484426-cd351280-1969-11eb-8ebb-be9ee146689c.png)


### Пример 2
Шумная | Денойзинг | Чистая

![noisy_2](https://user-images.githubusercontent.com/15848838/97484914-6fed9100-196a-11eb-9cf0-187cc617a376.png)

![denoised_2](https://user-images.githubusercontent.com/15848838/97484927-7419ae80-196a-11eb-84a2-3c3431c252f9.png)

![clean_2](https://user-images.githubusercontent.com/15848838/97484953-7d0a8000-196a-11eb-8642-1953741bddf8.png)

