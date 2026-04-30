import numpy as np
import pandas as pd
from flask import jsonify
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from supabase import create_client
import os
import unicodedata
import re

# ---------------- SUPABASE INIT ---------------- #
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------- HELPER: aggressive normalise ---------------- #
def normalise(text):
    """
    Strip whitespace, lowercase, remove all non-alphanumeric characters,
    and normalise unicode (e.g. accented chars, zero-width spaces).
    'Chhattisgarh ' -> 'chhattisgarh'
    'Andhra Pradesh' -> 'andhrapradesh'
    """
    text = str(text)
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.strip().lower()
    text = re.sub(r'[^a-z0-9]', '', text)
    return text


# ---------------- HELPER: paginated fetch (bypasses 1000-row Supabase limit) ---------------- #
def fetch_all_rows(table_name):
    """
    Supabase .select("*") returns max 1000 rows by default.
    This fetches all rows in pages of 1000 until the table is fully loaded.
    """
    all_rows = []
    page_size = 1000
    offset = 0

    while True:
        res = (
            supabase
            .table(table_name)
            .select("*")
            .range(offset, offset + page_size - 1)
            .execute()
        )

        if not res.data:
            break

        all_rows.extend(res.data)
        print(f"  📦 Fetched {len(all_rows)} rows from '{table_name}' so far...")

        # If we got fewer rows than page_size, we've hit the end
        if len(res.data) < page_size:
            break

        offset += page_size

    print(f"  ✅ Total rows fetched from '{table_name}': {len(all_rows)}")
    return all_rows


# ---------------- CROP RECOMMENDATION ---------------- #
class CropService:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.id_map = {}

    def load_data(self):
        rows = fetch_all_rows("crop_recommendation")

        if not rows:
            raise ValueError("❌ crop_recommendation table is empty")

        df = pd.DataFrame(rows)

        print("📊 Crop data loaded:", df.shape)
        print("Columns:", df.columns.tolist())

        df = df.dropna()

        if df.empty:
            raise ValueError("❌ All rows contain NULL values")

        if 'label' not in df.columns:
            raise ValueError("❌ Column 'label' missing")

        return df

    def train(self):
        df = self.load_data()

        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

        y = self.label_encoder.fit_transform(df['label'])

        self.id_map = {
            i: label for i, label in enumerate(self.label_encoder.classes_)
        }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=43
        )

        self.model.fit(X_train, y_train)

        acc = self.model.score(X_test, y_test)
        print(f"✅ Crop model trained. Accuracy: {acc:.2f}")

    def predict(self, data):
        if self.model is None:
            raise ValueError("❌ Crop model not trained")

        try:
            x = np.array([[
                float(data['N']),
                float(data['P']),
                float(data['K']),
                float(data['temperature']),
                float(data['humidity']),
                float(data['ph']),
                float(data['rainfall'])
            ]])

            pred = int(self.model.predict(x)[0])
            prob = float(max(self.model.predict_proba(x)[0]))

            return {
                "crop": self.id_map[pred],
                "confidence": prob
            }

        except KeyError as e:
            raise ValueError(f"Missing input field: {str(e)}")


# ----------------------------------------------------
# ---------------- YIELD PREDICTION ---------------- #
# ----------------------------------------------------
class YieldService:
    def __init__(self):
        self.model = None
        self.state_map = {}
        self.crop_map  = {}

    def load_data(self):
        rows = fetch_all_rows("crop_yield")

        if not rows:
            raise ValueError("❌ crop_yield table is empty")

        df = pd.DataFrame(rows)

        print("📊 Yield data loaded:", df.shape)

        # Rename columns with spaces
        df.columns = df.columns.str.replace(' ', '_', regex=False)

        return df

    def train(self):
        df = self.load_data()

        # Build normalised maps — now covers ALL rows, not just first 1000
        self.state_map = {
            normalise(s): i for i, s in enumerate(df['State_Name'].unique())
        }

        self.crop_map = {
            normalise(c): i for i, c in enumerate(df['Crop'].unique())
        }

        print("✅ State map keys:", list(self.state_map.keys()))
        print("✅ Crop map keys:", list(self.crop_map.keys()))

        df['state_code'] = df['State_Name'].apply(normalise).map(self.state_map)
        df['crop_code']  = df['Crop'].apply(normalise).map(self.crop_map)

        X = df[
            [
                'state_code',
                'crop_code',
                'Area_ha',
                'Total_N_kg',
                'Total_P_kg',
                'Total_K_kg',
                'Temperature_C',
                'Humidity_%',
                'pH',
                'Rainfall_mm'
            ]
        ]

        y = df['Yield_kg_per_ha']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        self.model.fit(X_train, y_train)

        score = self.model.score(X_test, y_test)
        print(f"✅ Yield model trained. R² score: {score:.2f}")

    def predict(self, data):
        if self.model is None:
            raise ValueError("❌ Yield model not trained")

        try:
            state_key = normalise(data['state_name'])
            crop_key  = normalise(data['crop'])

            print(f"🔍 Normalised state key: '{state_key}'")
            print(f"🔍 Normalised crop key:  '{crop_key}'")

            state_code = self.state_map.get(state_key)
            if state_code is None:
                raise ValueError(
                    f"Unknown state: '{data['state_name']}' "
                    f"(normalised: '{state_key}'). "
                    f"Available: {list(self.state_map.keys())}"
                )

            crop_code = self.crop_map.get(crop_key)
            if crop_code is None:
                raise ValueError(
                    f"Unknown crop: '{data['crop']}' "
                    f"(normalised: '{crop_key}'). "
                    f"Available: {list(self.crop_map.keys())}"
                )

            x = [[
                state_code,
                crop_code,
                float(data['area_ha']),
                float(data.get('total_n_kg', 0)),
                float(data.get('total_p_kg', 0)),
                float(data.get('total_k_kg', 0)),
                float(data['temperature']),
                float(data['humidity']),
                float(data['ph']),
                float(data['rainfall'])
            ]]

            pred = self.model.predict(x)[0]
            return max(0, float(pred))

        except KeyError as e:
            raise ValueError(f"Missing field: {str(e)}")


# ---------------- GLOBAL INSTANCES ---------------- #
crop_service = None
yield_service = None


# ---------------- REGISTER ROUTES ---------------- #
def register_ml_routes(app):
    global crop_service, yield_service

    try:
        print("🚀 Training ML models...")

        crop_service = CropService()
        crop_service.train()

        yield_service = YieldService()
        yield_service.train()

        print("✅ ML models ready")

    except Exception as e:
        print("❌ ML TRAINING FAILED:", str(e))
        crop_service = None
        yield_service = None