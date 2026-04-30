FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY "class_names(foodsg).json" .
COPY keras/model-10-0.64.keras ./model.keras

EXPOSE 8000

CMD sh -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"
