"""Shared inference logic used by both the FastAPI service and the Streamlit app."""
import os
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"

HOTELS = ["City Hotel", "Resort Hotel"]
MEALS = ["BB", "FB", "HB", "SC"]
COUNTRIES = ["PRT", "GBR", "FRA", "ESP", "DEU", "ITA", "IRL", "BRA", "NLD", "Other"]
MARKET_SEGMENTS = ["Aviation", "Complementary", "Corporate", "Direct", "Groups", "Offline TA/TO", "Online TA"]
DISTRIBUTION_CHANNELS = ["Corporate", "Direct", "GDS", "TA/TO"]
ROOM_TYPES = ["A", "B", "C", "D", "E", "F", "G", "H", "L"]
DEPOSIT_TYPES = ["No Deposit", "Non Refund", "Refundable"]
CUSTOMER_TYPES = ["Contract", "Group", "Transient", "Transient-Party"]

CATEGORICAL = ["hotel", "meal", "country", "market_segment", "distribution_channel",
               "reserved_room_type", "deposit_type", "customer_type"]


def risk_level(prob: float) -> str:
    return "Low" if prob < 0.3 else "Medium" if prob < 0.6 else "High"


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model = joblib.load(model_dir / "lgbm_model.pkl")
        self.columns = joblib.load(model_dir / "feature_columns.pkl")
        self.threshold = float(joblib.load(model_dir / "threshold.pkl"))

    def build_features(self, booking: dict) -> pd.DataFrame:
        """Turn one raw booking into the one-hot encoded row the model was trained on."""
        b = dict(booking)
        arrival: date = b.pop("arrival_date")
        has_agent = int(b.pop("has_agent", False))
        row = {
            **b,
            "arrival_date_week_number": arrival.isocalendar()[1],
            "arrival_date_day_of_month": arrival.day,
            "arrival_month": arrival.month,
            "total_nights": b["stays_in_weekend_nights"] + b["stays_in_week_nights"],
            "total_guests": b["adults"] + b["children"] + b["babies"],
            "has_kids": int(b["children"] + b["babies"] > 0),
            "has_agent": has_agent,
            "ever_canceled_before": int(b["previous_cancellations"] > 0),
        }
        df = pd.get_dummies(pd.DataFrame([row]), columns=CATEGORICAL)
        # drop_first during training removed one level per column; reindex drops it here too
        return df.reindex(columns=self.columns, fill_value=0).astype(float)

    def predict(self, booking: dict, top_n: int = 5) -> dict:
        X = self.build_features(booking)
        prob = float(self.model.predict_proba(X)[0, 1])
        # LightGBM's built-in SHAP-style contributions; last column is the bias term
        contrib = self.model.predict(X, pred_contrib=True)[0][:-1]
        factors = sorted(
            ({"feature": f, "value": float(X.iloc[0][f]), "impact": float(c)}
             for f, c in zip(self.columns, contrib)),
            key=lambda d: abs(d["impact"]), reverse=True,
        )[:top_n]
        for f in factors:
            f["direction"] = "increases" if f["impact"] > 0 else "decreases"
        return {
            "cancel_probability": round(prob, 4),
            "will_cancel": prob >= self.threshold,
            "risk_level": risk_level(prob),
            "threshold": self.threshold,
            "top_factors": factors,
        }


def gemini_summary(prob: float, factors: list[dict]) -> str:
    """Plain-English note for front-desk staff. Needs Google_api_key in .env."""
    from dotenv import load_dotenv
    from google import genai

    load_dotenv()
    api_key = os.getenv("Google_api_key")
    if not api_key:
        raise RuntimeError("Google_api_key is not set in .env")
    reasons = "\n".join(f"- {f['feature']} = {f['value']:g} ({f['direction']} cancellation risk)" for f in factors)
    prompt = f"""You are an assistant for hotel front-desk staff.
A machine learning model predicted this booking's cancellation probability as {prob:.0%}.
The main factors behind this prediction are:
{reasons}

Feature notes: lead_time = days between booking and arrival.
One-hot columns like market_segment_Online TA = 1 mean the booking belongs to that category;
0 means it does not.

Write a short note for hotel staff (3-4 sentences, plain English, no technical terms):
1. State the risk level (Low / Medium / High).
2. Explain the main reasons simply.
3. Suggest one or two practical actions the hotel can take."""
    client = genai.Client(api_key=api_key)
    return client.models.generate_content(model="gemini-2.5-flash", contents=prompt).text
