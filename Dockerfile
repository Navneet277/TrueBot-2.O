   FROM python:3.11-slim

   RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

   WORKDIR /app

   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt

   COPY . .

   RUN python -m nltk.downloader stopwords wordnet punkt && \
       python scripts/prepare_dataset.py && \
       python -m modules.train_model && \
       python scripts/init_db.py

   EXPOSE 5000

   CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]