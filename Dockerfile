FROM python:3.13-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    AVITO_PROJECT_DIR=/work

WORKDIR /work

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt

# Paddle устанавливается из официального репозитория; остальные версии берутся из requirements.txt.
RUN python -m pip install paddlepaddle==3.3.0 \
      -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
    && python -m pip install -r /tmp/requirements.txt \
    && python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"

# Выполняем ячейки напрямую, чтобы прогресс и замеры сразу появлялись в консоли Docker.
ENTRYPOINT ["python", "-u", "-c", "import json; from pathlib import Path; cells=json.loads(Path('main.ipynb').read_text(encoding='utf-8'))['cells']; scope={'__name__':'__main__'}; [exec(compile(''.join(cell['source']), f'main.ipynb cell {i}', 'exec'), scope) for i, cell in enumerate(cells, 1) if cell['cell_type']=='code']"]
