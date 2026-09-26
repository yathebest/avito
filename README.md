# Ориентация текстового кропа для Avito

Для каждого изображения предсказывается p_180 — вероятность, что текст перевёрнут на 180°. Результат — submission.csv с колонками image_id,p_180 и 20 000 строками. Оценка: 1 − mean((p_180 − y_180)^2).

## Итоговое решение

Финальный submission.csv создан скриптом predict_submission.py с локальными весами [PP-LCNet_x1_0_textline_ori](https://paddlepaddle.github.io/PaddleX/3.7/en/module_usage/tutorials/ocr_modules/textline_orientation_classification.html) из PaddleOCR/PaddleX. Каждое изображение подаётся модели дважды: как есть и после точного поворота на 180°. Если модель выдаёт вероятности переворота p0 и p1, итог равен sigmoid((logit(p0) − logit(p1)) / 2). Перед logit значения ограничиваются интервалом [1e-6, 1-1e-6]. Коэффициент 1/2 фиксирован; дополнительная модель не обучается.

Веса лежат в models/PP-LCNet_x1_0_textline_ori_infer/ (около 6,5 МБ). При инференсе используется одна компактная модель, но два её прогона на кроп. Внешние API и LLM/VLM не используются.

## Запуск

Проверено на Python 3.13.9, PaddleOCR 3.7.0, PaddlePaddle 3.3.0. Из корня проекта:

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    $env:AVITO_TEST_PATH = 'C:\path\to\test_data'
    .\.venv\Scripts\python.exe predict_submission.py

Вместо распакованной папки можно передать путь к test.zip. В папке ожидаются sample_submission.csv и test/images/; в архиве — sample_submission.csv и test/images/*.png. Если AVITO_TEST_PATH не задана, скрипт ищет test.zip или test_data/ рядом с собой, затем в Downloads. Выход пишется в submission.csv рядом со скриптом. Скрипт проверяет количество, уникальность и наличие всех тестовых изображений. Запуск на тех же данных с теми же локальными весами воспроизводит отправленный CSV.

## Проверка качества

Для подбора подхода использовался отдельный официальный split [HierText validation](https://github.com/google-research-datasets/hiertext): 3000 исходных кропов и их точные повороты на 180° (всего 6000). В HierText нет готовых меток ориентации. Исходно прямыми считаются строки минимум из трёх слов с порядком чтения слева направо и углом не больше 10°. Это геометрическая псевдоразметка, возможны ошибки. Разделение train/validation сделано по официальным исходным изображениям, а не по случайным кропам. Тест Avito вручную не размечался и не использовался для обучения.

| Подход | 1 − Brier на HierText validation |
| --- | ---: |
| PaddleOCR, один прогон | 0.93908 |
| MobileNetV3-Small после обучения на синтетике и HierText train | 0.92319 |
| Ансамбль MobileNet и однопроходной PaddleOCR | 0.95373 |
| Итоговый PaddleOCR с двумя поворотами | **0.98424** |

Обе оценки PaddleOCR можно воспроизвести по сохранённым предсказаниям: python validation/evaluate.py. validation/prepare_validation.py создаёт валидационную выборку из исходного HierText; evaluate_model.py повторно получает предсказания модели, если исходные изображения скачаны. Случайная выборка фиксирована seed 20260926. Файлы экспериментов с MobileNet и промежуточными метриками лежат в models/; они не участвуют в финальном инференсе. Выбрана одна PaddleOCR, потому что два поворота дали лучший результат на этой валидации при меньшем размере, чем ансамбль.

Эти числа не равны скрытому скору Avito. Отправленный submission.csv получил на Stepik **0.95944706**; это единственный официальный замер данного решения, и он показывает разницу между HierText и тестом Avito.

## Повторение экспериментов

Для обучения MobileNet используйте отдельное виртуальное окружение с requirements-train.txt: в нём закреплены версии PyTorch 2.6.0 и TorchVision 0.21.0, применённые в эксперименте. Для GPU выберите подходящую сборку PyTorch по [официальной инструкции](https://pytorch.org/get-started/previous-versions/). Скачайте [HierText](https://github.com/google-research-datasets/hiertext): old_train_data/json/train.jsonl и JPG из train.tgz в old_train_data/, а old_validation_data/validation.jsonl.gz и JPG в old_validation_data/validation/. Затем запустите:

    python make_hiertext_crops.py
    python build_train_orientation_csv.py
    python make_synthetic_orientation.py
    python train_synthetic_orientation.py
    python prepare_real_orientation.py
    python train_real_orientation.py
    python evaluate_real_orientation.py

make_synthetic_orientation.py использует системные шрифты Windows; для другого набора шрифтов можно задать AVITO_FONT_DIR. Обучающие скрипты используют фиксированные seed. Данные и чекпоинты MobileNet не требуются для получения итогового submission.csv.

Использованы открытые [PaddleOCR/PaddleX](https://github.com/PaddlePaddle/PaddleOCR), [PyTorch/TorchVision](https://pytorch.org/) для экспериментов, [HierText](https://github.com/google-research-datasets/hiertext), NumPy и Pillow. Локальные веса PaddleOCR взяты из открытой модели, указанной выше.