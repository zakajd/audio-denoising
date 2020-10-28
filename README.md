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
* Загрузить веса моделей [audio_classification_weights.zip](https://drive.google.com/file/d/1UEjq1AO8C7GSpC0e6WPtnqMfplab0-op/view?usp=sharing) [audio_denoising_weights.zip](https://drive.google.com/file/d/1MheJsOTZ_9hmCQGJ6gRC3lOHXIS1aIPf/view?usp=sharing) в папку проекта
* Разархивировать и переместить их в папки с логами и параметрами моделей: `unzip audio_classification_weights.zip && mv model.chpn logs/classification_best` `unzip audio_denoising_weights.zip && mv model.chpn logs/denoising_best`
* Классификация `python3 test.py --help`. 
* Денойзинг `python3 test.py --config_path logs/denoising_best --file_path examples/noisy/82_121544_82-121544-0008.npy`

### Обучение модели
* Отредактируйте файл настроек `configs/default.yaml` указав путь до папки с данными.
* Классификация `CUDA_VISIBLE_DEVICES=0 python3 train.py training=classification`
* Денойзинг `CUDA_VISIBLE_DEVICES=0 python3 train.py training=denoising`

### Описание решения
Для задачи классификации была использована классическая архитектура ResNet50. Первые эксперименты показали, что из-за искусственного характера наложенного шума, уже за 1 эпоху предобученная сеть получает точность 100% и дальнейшее обучение не имеет смысла.

Для задачи деноизинга была использована архитектура типа Unet с предобученным энкодером SeResNet50. Обучение с MSE в качестве лосса дало сглаженный результат, см. пример 1. Дообучение с L1 лоссом улучшило результат визуально, но сделало хуже по метрике, см. пример 2.