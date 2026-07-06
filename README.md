# MindGuard AI

Text-based mental-health risk screening that runs entirely on-device. Paste in
how you've been feeling and MindGuard classifies the input as **Normal**,
**Stress**, **Anxiety** or **Depression** — with the actual keywords that
drove the prediction surfaced for transparency.

![status](https://img.shields.io/badge/status-prototype-blue)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![ui](https://img.shields.io/badge/UI-Streamlit-red)
![ml](https://img.shields.io/badge/ML-XGBoost%20%2B%20scikit--learn-orange)

> Not a diagnostic tool. MindGuard is an awareness / self-screening aid; it
> is not a substitute for a qualified mental-health professional. The UI
> includes an explicit disclaimer and a Resource Hub of helplines.

## Features

- **Four-class risk screening** — Normal, Stress, Anxiety, Depression — from
  free-form text.
- **Explainable predictions** — the model surfaces the top influential
  keywords for every classification (from XGBoost feature importance and
  Logistic-Regression coefficients).
- **Side-by-side model comparison** — a Logistic-Regression baseline trained
  alongside the XGBoost model with shared confusion matrices and metrics.
- **Voice input** — optional browser-side Web Speech API capture (Chrome
  recommended) so you can speak instead of type.
- **Auto-training on first run** — if the pickled model isn't present,
  MindGuard trains it from the bundled dataset before the UI becomes
  interactive (one-time, ~1-2 minutes).
- **Privacy first** — no data leaves the host. The model and vectoriser are
  loaded locally and inference runs in-process.

## Tech stack

| Layer         | Library                                            |
|---------------|----------------------------------------------------|
| GUI           | Streamlit                                          |
| Text pre-proc | NLTK (stopwords + WordNet lemmatiser), regex       |
| Vectoriser    | scikit-learn TfidfVectorizer (5 000 features, 1-2 grams) |
| Classifiers   | XGBoost (primary), LogisticRegression (baseline)   |
| Persistence   | joblib (`.pkl` models in `models/`)                |
| Voice         | Web Speech API (browser) + SpeechRecognition (offline diag) |

No transformers, no PyTorch — deliberately light so it runs on a laptop with
no GPU.

## Architecture

```
text input ──► preprocess.clean_text() ──► TfidfVectorizer ──► XGBoost ──► label
                                                            │
                                                            └──► top-5 keywords
```

- `app.py` — Streamlit UI: Analyze / Model Comparison / History / About tabs.
- `src/preprocess.py` — `clean_text()`: regex clean, stopword removal,
  WordNet lemmatisation.
- `src/model.py` — `train_model()` and `predict_risk()`. Trains XGBoost and
  the Logistic-Regression baseline together so they can be compared.
- `models/` — serialised `mental_health_model.pkl`, `baseline_model.pkl`,
  `tfidf_vectorizer.pkl`, `label_encoder.pkl`, `model_comparison.pkl`.
- `data/` — bundled mental-health text dataset plus synthetic samples used
  on the first training run.
- `Dockerfile`, `Procfile`, `packages.txt` — Heroku / Railway / Cloud Run
  ready deployment.

## Installation

```bash
git clone https://github.com/iamkk2307/MindGuard-AI.git
cd MindGuard-AI
pip install -r requirements.txt
streamlit run app.py
```

The first launch trains the model from the bundled CSV (and synthetic
augmentation), writes the artefacts to `models/`, and then loads the
interactive UI. Subsequent launches are instant.

## Docker

```bash
docker build -t mindguard .
docker run -p 8501:8501 mindguard
```

## Cloud deployment

The repo ships with both a `Procfile` and a `Dockerfile`:

- **Heroku / Railway**: connect the repo. The `Procfile`
  (`web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`)
  handles the rest.
- **Google Cloud Run / AWS ECS**: build the Docker image and deploy as a
  service exposing port 8501.

## Voice input

The voice button opens a small popup window that calls the browser's Web
Speech API directly — bypassing Streamlit's iframe restrictions. This means
no audio bytes are ever sent over the network; the transcription happens
client-side. Currently best-supported on Chrome / Edge / Chromium.

## Roadmap

- [ ] Replace TF-IDF with a small sentence-transformer for richer context.
- [ ] Multilingual support (current model is English-only).
- [ ] LLM-augmented synthetic-sample generation.
- [ ] CI / automated test suite (currently only manual diagnostic scripts).
- [ ] Remove unused dependencies (`pywebview`, `streamlit-audiorecorder`).

## License

MIT
