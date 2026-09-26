# Ориентация текстовых кропов Avito

Нужно предсказать `p_180` — вероятность, что текст в кропе перевёрнут на 180°. Каждый ноутбук самостоятельно читает 20 000 изображений, сверяет их с `sample_submission.csv`, делает предсказания и записывает `submission.csv` с колонками `image_id,p_180`. Время загрузки моделей, чтения изображений, инференса и всего запуска выводится в конце. В проекте нет отдельных Python-скриптов, необходимых для работы ноутбуков.

## Два варианта

| Ноутбук | Модели | Устройство | Проверка на HierText validation |
| --- | --- | --- | ---: |
| `main.ipynb` | [PP-LCNet_x1_0_textline_ori](https://paddlepaddle.github.io/PaddleX/3.7/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html), исходный кроп и поворот на 180° | CPU | 0.98424 по 1−Brier |
| `main_full_ocr.ipynb` | та же модель и [eslav_PP-OCRv5_mobile_rec](https://paddlepaddle.github.io/PaddleX/3.7/en/module_usage/tutorials/ocr_modules/text_recognition.html), обе ориентации каждого кропа | GPU при наличии CUDA, иначе CPU | 0.99409 по 1−Brier |

Быстрый вариант объединяет две оценки классификатора как `sigmoid((logit(p0) − logit(p180)) / 2)`. Полный вариант добавляет к его logit разницу уверенности распознавания текста: `20 × (score_180 − score_0)`. В нём OCR вызывается для **каждого** изображения, без отбора по уверенности. Веса обеих моделей лежат в `models/`; во время инференса ничего не скачивается и внешние API не используются.

Быстрый вариант дал **0.95944706** на Stepik. Полный вариант на Stepik пока не проверен; значение 0.99409 получено только на псевдоразмеченной валидации [HierText](https://github.com/google-research-datasets/hiertext) и не гарантирует такой же результат на тесте Avito. Тестовые изображения вручную не размечались. Из 3000 исходных строк HierText сделаны версии с поворотом на 180°; исходную ориентацию определяли по геометрии строк. Сохранённые данные этой проверки находятся в `validation/`.

Быстрый ноутбук дополнительно записывает `submission_hard.csv`: значение 1 при итоговом `p_180 > 0.5`, иначе 0. Порог применяется **после объединения двух поворотов**, а не к уверенности одного вызова классификатора. На 6000 валидационных кропов результаты по `1−Brier`: вероятности — **0.98424**, жёсткий порог 0.5 — **0.97933**, порог 0.65 — **0.97667**. На всех пяти частях разбиения по 964 исходным изображениям вероятности оказались лучше жёсткой классификации. Поэтому основным файлом для отправки остаётся вероятностный `submission.csv`; пороговый файл нужен для сравнения. Порог 0.5 выбран как симметричная граница двух ориентаций: подбор максимума на всей валидации дал 0.512 и только 0.97967, что не оправдывает подстройку под эту выборку. По умолчанию оба CSV пишутся в корень проекта; переменная `AVITO_OUTPUT_DIR` позволяет указать другую папку, не заменяя существующий `submission.csv`.

## Запуск

1. Откройте нужный `.ipynb` из корня проекта в Jupyter или VS Code и выберите Python-окружение с PaddleOCR 3.7.0, NumPy 2.3.5 и Pillow 12.1.0. Для быстрого CPU-варианта зависимости перечислены в `requirements.txt`. Для GPU-варианта установите подходящую [CUDA-сборку PaddlePaddle 3.3.0](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/windows-pip.html) вместо CPU-сборки.
2. Положите распакованный тест в `test_data/` (`sample_submission.csv` и `test/images/*.png`) или `test.zip` в корень проекта. Если файлы лежат в другом месте, задайте `AVITO_TEST_PATH`; если Jupyter открыт из другой папки, задайте `AVITO_PROJECT_DIR`.
3. Выполните **Run All**. Ноутбук создаст или заменит `submission.csv` в корне проекта и напечатает фактическое время на вашем устройстве.

Запуск второго ноутбука заменит CSV от первого. Для сравнения сохраните копию нужного файла до следующего запуска. Ноутбуки используют локальные компактные модели; LLM/VLM не применяются.

## Запуск в Docker

Проверяющему не нужно обучать модель: веса находятся в репозитории в `models/`. Нужны Docker и выданный тестовый набор в `test_data/` или `test.zip`; затем достаточно двух команд ниже. Контейнер запускает те же ноутбуки без установки PaddleOCR в системный Python. Перейдите в корень проекта в PowerShell. Проект монтируется в `/work`, поэтому полученный `submission.csv` сохраняется на компьютере рядом с ноутбуками. Выполненный ноутбук с замерами времени сохраняется в `results/executed.ipynb`.

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

`requirements.txt` нужен только для запуска без Docker. Версии основных библиотек зафиксированы и в Dockerfile. Образ не содержит тестовые данные и веса: они читаются из примонтированного проекта. Веса уже хранятся в `models/` репозитория; проверяющему нужен только выданный ему тестовый набор.

## Kaggle: полный OCR на GPU

Docker внутри Kaggle Notebook не нужен. Добавьте приватный Kaggle Dataset с папками `models/` и `test_data/` в той же структуре, что в этом проекте. В настройках Notebook выберите GPU и включите Internet на время установки пакетов. **Не используйте `pip install -r requirements.txt` для GPU:** там указана CPU-сборка Paddle. Перед первой ячейкой `main_full_ocr.ipynb` добавьте две ячейки:

```python
%pip install --no-cache-dir paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
%pip install --no-cache-dir paddleocr==3.7.0 numpy==2.3.5 Pillow==12.1.0
```

```python
import os
from pathlib import Path

data_dir = Path("/kaggle/input/ИМЯ-ВАШЕГО-ДАТАСЕТА")
assert (data_dir / "models" / "PP-LCNet_x1_0_textline_ori_infer").is_dir()
assert (data_dir / "models" / "eslav_PP-OCRv5_mobile_rec_infer").is_dir()
assert (data_dir / "test_data" / "test" / "images").is_dir()
os.environ["AVITO_PROJECT_DIR"] = str(data_dir)
os.environ["AVITO_OUTPUT_DIR"] = "/kaggle/working"
os.environ["AVITO_REQUIRE_GPU"] = "1"
```

Затем выполните **Run All**. Итоговый файл будет в `/kaggle/working/submission.csv`. В выводе первой ячейки ноутбука проверьте `CUDA build: True`, `visible GPUs` больше нуля и `inference device: gpu:0`. Если на выделенном Kaggle GPU установленная CUDA-сборка Paddle не работает, ноутбук остановится до обработки 20 000 файлов; уберите `AVITO_REQUIRE_GPU` для CPU-варианта или выберите совместимую сборку Paddle согласно [официальной инструкции](https://www.paddlepaddle.org.cn/documentation/docs/en/install/pip/linux-pip_en.html). После установки зависимостей распознавание использует локальные веса и не обращается к внешним API.
