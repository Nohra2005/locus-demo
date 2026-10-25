FROM python:3.11-slim
WORKDIR /app
COPY gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY gateway/ .
# frontend/dist is provided at runtime via docker-compose volume mount
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
