FROM python:3.11

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt
ENV PYTHONPATH=/app

EXPOSE 8881
CMD ["bash", "-c", "python app/sleep_ai/resources/RAG/qdrant_loader.py && uvicorn main:app --host 0.0.0.0 --port 8881"]