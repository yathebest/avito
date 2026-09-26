# Ориентация текстовых кропов Avito

Нужно предсказать `p_180` — вероятность, что текст в кропе перевёрнут на 180°. Каждый ноутбук самостоятельно читает 20 000 изображений, сверяет их с `sample_submission.csv`, делает предсказания и записывает `submission.csv` с колонками `image_id,p_180`. Время загрузки моделей, чтения изображений, инференса и всего запуска выводится в конце. В проекте нет отдельных Python-скриптов, необходимых для работы ноутбуков.

## Два варианта

| Ноутбук | Модели | Устройство | Проверка на HierText validation |
| --- | --- | --- | ---: |
| `main.ipynb` | [PP-LCNet_x1_0_textline_ori](https://paddlepaddle.github.io/PaddleX/3.7/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html), исходный кроп и поворот на 180° | CPU | 0.98424 по 1−Brier |
| `main_full_ocr.ipynb` | та же модель и [eslav_PP-OCRv5_mobile_rec](https://paddlepaddle.github.io/PaddleX/3.7/en/module_usage/tutorials/ocr_modules/text_recognition.html), обе ориентации каждого кропа | GPU при наличии CUDA, иначе CPU | 0.99409 по 1−Brier |

Быстрый вариант объединяет две оценки классификатора как `sigmoid((logit(p0) − logit(p180)) / 2)`. Полный вариант добавляет к его logit разницу уверенности распознавания текста: `20 × (score_180 − score_0)`. В нём OCR вызывается для **каждого** изображения, без отбора по уверенности. Веса обеих моделей лежат в `models/`; во время инференса ничего не скачивается и внешние API не используются.

Быстрый вариант дал **0.95944706** на Stepik. Полный вариант на Stepik пока не проверен; значение 0.99409 получено только на псевдоразмеченной валидации [HierText](https://github.com/google-research-datasets/hiertext) и не гарантирует такой же результат на тесте Avito. Тестовые изображения вручную не размечались. Из 3000 исходных строк HierText сделаны версии с поворотом на 180°; исходную ориентацию определяли по геометрии строк. Сохранённые данные этой проверки находятся в `validation/`.

## Запуск

1. Откройте нужный `.ipynb` из корня проекта в Jupyter или VS Code и выберите Python-окружение с PaddleOCR 3.7.0, NumPy 2.3.5 и Pillow 12.1.0. Для быстрого CPU-варианта зависимости перечислены в `requirements.txt`. Для GPU-варианта установите подходящую [CUDA-сборку PaddlePaddle 3.3.0](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip.html) вместо CPU-сборки.
2. Положите распакованный тест в `test_data/` (`sample_submission.csv` и `test/images/*.png`) или `test.zip` в корень проекта. Если файлы лежат в другом месте, задайте `AVITO_TEST_PATH`; если Jupyter открыт из другой папки, задайте `AVITO_PROJECT_DIR`.
3. Выполните **Run All**. Ноутбук создаст или заменит `submission.csv` в корне проекта и напечатает фактическое время на вашем устройстве.

Запуск второго ноутбука заменит CSV от первого. Для сравнения сохраните копию нужного файла до следующего запуска. Ноутбуки используют локальные компактные модели; LLM/VLM не применяются.

## Запуск в Docker

Контейнер запускает те же ноутбуки без установки PaddleOCR в системный Python. Перейдите в корень проекта в PowerShell. Папки `models/` и `test_data/` должны быть здесь же: проект монтируется в `/work`, поэтому полученный `submission.csv` сохраняется на компьютере рядом с ноутбуками. Выполненный ноутбук с замерами времени сохраняется в `results/executed.ipynb`.

Быстрый CPU-вариант:

```powershell
docker build -t avito-orientation:cpu .
docker run --rm --network none --mount "type=bind,source=$($PWD.Path),target=/work" avito-orientation:cpu main.ipynb
```

Полный вариант на NVIDIA GPU:

```powershell
docker build --build-arg PADDLE_VARIANT=gpu -t avito-orientation:gpu .
docker run --rm --network none --gpus all --mount "type=bind,source=$($PWD.Path),target=/work" avito-orientation:gpu main_full_ocr.ipynb
```

Для второго варианта нужны совместимые драйвер NVIDIA, Docker с доступом к GPU и достаточно памяти видеокарты; на Windows Docker Desktop использует WSL 2. Если GPU недоступен, `main_full_ocr.ipynb` можно выполнить в CPU-контейнере, заменив имя ноутбука в первой команде `docker run`. Проверить установленный внутри образа Paddle можно так:

```powershell
docker run --rm --entrypoint python avito-orientation:cpu -c "import paddle, paddleocr; print(paddle.__version__, paddle.is_compiled_with_cuda())"
```

На Linux запускайте те же команды из Bash, заменив `source=$($PWD.Path)` на `source=$(pwd)`. CPU-образ рассчитан на x86-64 с поддержкой AVX. GPU-образ собирается с CUDA 12.9 и требует совместимого NVIDIA-драйвера. При запуске сеть контейнера отключена (`--network none`): инференс использует только локальные веса.

`requirements.txt` нужен только для запуска без Docker. Версии основных библиотек зафиксированы и в Dockerfile. Образ не содержит тестовые данные и веса: они читаются из примонтированного проекта, поэтому передача результатов проверяющему требует приложить `models/` и исходные тестовые файлы либо указать, где их получить.
