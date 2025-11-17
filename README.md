# TrueBot – Fake News Detection System

TrueBot is an end-to-end AI system that classifies news content as Real or Fake using TF-IDF vectorization, ensemble machine learning, and a curated dataset that mirrors common misinformation patterns.

## Features

- Modular preprocessing, training, and prediction pipelines
- Flask + Bootstrap 5 web UI focused on manual text entry
- SQLite persistence for classroom experiments and manual checks
- Automated training script with Logistic Regression, Random Forest, and optional XGBoost comparison
- Unit and integration tests with pytest

## Project Structure

```
TrueBot/
├── app.py
├── config.py
├── data/news.csv
├── database/truebot.db
├── model/
├── modules/
├── static/
├── templates/
└── tests/
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader stopwords wordnet punkt
python -m spacy download en_core_web_sm  # optional
python scripts/prepare_dataset.py        # builds the large synthetic dataset
python -m modules.train_model            # generates model/model.pkl and vectorizer.pkl
python app.py
```

Visit `http://localhost:5000`.

## Testing

```bash
pytest
python tests/evaluate_model.py
```

## Deployment

- Use `gunicorn app:app` on Render/Railway.
- Set environment variables:
  - `SECRET_KEY`
  - `PORT` (if required by platform)
  - `DATABASE_URL` (optional – can stay SQLite)

Run migrations by calling `python -c "from app import init_db; init_db()"`.


