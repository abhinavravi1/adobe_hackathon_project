FROM --platform=linux/amd64 python:3.10-slim
WORKDIR /app

COPY main.py .

RUN pip install --no-cache-dir sentence-transformers PyMuPDF

CMD ["python", "main.py"]