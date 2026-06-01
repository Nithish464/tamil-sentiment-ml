FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/     ./src/
COPY api/     ./api/
COPY models/  ./models/
COPY dashboard/ ./dashboard/
EXPOSE 7860
CMD ["python", "dashboard/app.py"]
