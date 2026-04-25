FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install .

RUN addgroup --system --gid 10001 app && adduser --system --uid 10001 --ingroup app app \
    && mkdir -p /app/data \
    && chown -R app:app /app

USER app

EXPOSE 8000

ENTRYPOINT ["bm-agent"]
CMD ["api", "--host", "0.0.0.0", "--port", "8000", "--flow", "data/flow.json"]
