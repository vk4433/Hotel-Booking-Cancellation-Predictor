# Hotel Booking Cancellation Predictor

Predicts whether a hotel booking will be cancelled (LightGBM), shows the factors behind the score,
and can write a plain-English note for front-desk staff with Gemini.

## Project layout

```
data/              raw and cleaned CSVs
notebooks/         01 exploration -> 02 cleaning & features -> 03 training -> 04 explainability
models/            lgbm_model.pkl, feature_columns.pkl, threshold.pkl (created by notebook 03)
src/predictor.py   inference code (feature building, prediction, Gemini summary)
streamlit_app.py   Streamlit web app
```

## Setup

```bash
pip install -r requirements.txt
```

For Gemini summaries, create a `.env` file: `Google_api_key=<your key>` (optional).

## Run

```bash
streamlit run streamlit_app.py   # http://localhost:8501
```

## Deploy to Streamlit Community Cloud

1. Push the repo to GitHub. `models/*.pkl` must be committed (about 22 MB). `data/` is not needed to run the app.
2. On https://share.streamlit.io choose **Create app**, pick the repo and branch, and set the main file to `streamlit_app.py`.
3. Optional, for Gemini summaries: **Advanced settings -> Secrets**, add `Google_api_key = "<your key>"`.
4. The pinned `scikit-learn` / `lightgbm` versions in `requirements.txt` must match the ones the model was trained with; retrain and update them together.

## Retraining

Run notebooks 02 then 03 in order; they overwrite `data/hotel_bookings_clean.csv` and `models/`.
