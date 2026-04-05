FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./
COPY alembic.ini ./
COPY alembic/ alembic/
COPY src/ src/

ENV UV_LINK_MODE=copy

RUN pip install uv && uv sync --no-dev --frozen

EXPOSE 8000

CMD ["sh", "-c", ".venv/bin/alembic upgrade head && .venv/bin/uvicorn openfable.main:app --host 0.0.0.0 --port 8000"]
