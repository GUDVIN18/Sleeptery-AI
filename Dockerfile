FROM python:3.12

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./requirements.st_bases.txt /requirements.st_bases.txt
RUN pip install --no-cache-dir -r /requirements.st_bases.txt

COPY . /app

ENV PYTHONPATH=/app

EXPOSE 8881
CMD ["bash", "-c", "python app/sleep_ai/resources/RAG/qdrant_loader.py && uvicorn main:app --host 0.0.0.0 --port 8881 --workers 1"]
