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

ENTRYPOINT ["jupyter", "nbconvert", "--execute", "--to", "notebook", "--ExecutePreprocessor.timeout=-1", "--output-dir=/work/results", "--output=executed.ipynb", "main.ipynb"]
