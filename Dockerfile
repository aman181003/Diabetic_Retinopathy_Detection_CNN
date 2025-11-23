# Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
ENV MODEL_PATH=/app/outputs/best_model.pth
EXPOSE 8080
CMD ["gunicorn","-b","0.0.0.0:8080","src.webapp.app:app","--workers","2"]
