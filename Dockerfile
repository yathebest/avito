FROM python:3.13-slim-bookworm

ARG PADDLE_VARIANT=cpu

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    AVITO_PROJECT_DIR=/work

WORKDIR /work

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Paddle provides separate CPU and CUDA wheels. The remaining Python packages
# are identical in both images; no system Python or Torch installation is used.
RUN case "$PADDLE_VARIANT" in \
      cpu) python -m pip install paddlepaddle==3.3.0 \
             -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ ;; \
      gpu) python -m pip install paddlepaddle-gpu==3.3.0 \
             -i https://www.paddlepaddle.org.cn/packages/stable/cu129/ ;; \
      *) echo "PADDLE_VARIANT must be cpu or gpu" >&2; exit 2 ;; \
    esac \
    && python -m pip install \
      paddleocr==3.7.0 \
      numpy==2.3.5 \
      Pillow==12.1.0 \
      nbconvert==7.16.6 \
      ipykernel==6.30.1 \
    && python -m ipykernel install --sys-prefix --name python3 --display-name "Python 3"

ENTRYPOINT ["jupyter", "nbconvert", "--execute", "--to", "notebook", "--ExecutePreprocessor.timeout=-1", "--output-dir=/work/results", "--output=executed.ipynb"]
CMD ["main.ipynb"]
