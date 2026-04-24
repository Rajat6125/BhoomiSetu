import numpy as np
import pandas as pd
from flask import request, jsonify
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from supabase import create_client
import os

# ---------------- SUPABASE INIT ---------------- #
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------- CROP RECOMMENDATION ---------------- #
class CropService:
    def __init__(self):
        self.model = None
        self.id_map = None

        # normalized lowercase columns
        self.features = [
            'n',
            'p',
            'k',
            'temperature',
            'humidity',
            'ph',
            'rainfall'
        ]

    def load_data(self):
        res = supabase.table("crop_recommendation").select("*").execute()
        df = pd.DataFrame(res.data)

        if df.empty:
            raise ValueError("crop_recommendation table is empty")

        # normalize column names
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(' ', '_', regex=False)
              .str.replace('%', '', regex=False)
        )

        print("Crop columns:", df.columns.tolist())

        if 'label' not in df.columns:
            raise ValueError("Column 'label' not found in crop_recommendation table")

        crops = sorted(df['label'].unique())

        self.id_map = {i: c for i, c in enumerate(crops)}
        reverse = {c: i for i, c in self.id_map.items()}

        df['label_id'] = df['label'].map(reverse)

        return df

    def train(self):
        df = self.load_data()

        X = df[self.features]
        y = df['label_id']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            eval_metric="mlogloss"
        )

        self.model.fit(X_train, y_train)

    def predict(self, data):
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


# ---------------- YIELD PREDICTION ---------------- #
class YieldService:
    def __init__(self):
        self.model = None
        self.state_map = {}
        self.crop_map = {}

    def load_data(self):
        res = supabase.table("crop_yield").select("*").execute()
        df = pd.DataFrame(res.data)

        if df.empty:
            raise ValueError("crop_yield table is empty")

        # normalize all columns
        df.columns = (
            df.columns
              .str.strip()
              .str.lower()
              .str.replace(' ', '_', regex=False)
              .str.replace('%', '', regex=False)
        )

        print("Yield columns:", df.columns.tolist())

        return df

    def train(self):
        df = self.load_data()

        self.state_map = {
            s: i for i, s in enumerate(df['state_name'].unique())
        }

        self.crop_map = {
            c: i for i, c in enumerate(df['crop'].unique())
        }

        df['state_code'] = df['state_name'].map(self.state_map)
        df['crop_code'] = df['crop'].map(self.crop_map)

        # Using your actual SQL column equivalents
        X = df[
            [
                'state_code',
                'crop_code',
                'area_ha',
                'total_n_kg',
                'total_p_kg',
                'total_k_kg',
                'temperature_c',
                'humidity_',
                'ph',
                'rainfall_mm'
            ]
        ]

        y = df['yield_kg_per_ha']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=42
        )

        self.model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        self.model.fit(X_train, y_train)

    def predict(self, data):

        x = [[
            self.state_map[data['state_name']],
            self.crop_map[data['crop']],
            float(data['area_ha']),
            float(data.get('N_req', 0)),
            float(data.get('P_req', 0)),
            float(data.get('K_req', 0)),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]]

        pred = self.model.predict(x)[0]

        return max(0, float(pred))


# ---------------- REGISTER ROUTES ---------------- #
def register_ml_routes(app):

    crop_service = CropService()
    yield_service = YieldService()

    print("🚀 Training ML models...")
    crop_service.train()
    yield_service.train()
    print("✅ ML models ready")


    @app.route('/predict', methods=['POST'])
    def predict_crop():
        data = request.get_json()
        result = crop_service.predict(data)
        return jsonify(result)


    @app.route('/predict_yield', methods=['POST'])
    def predict_yield():
        data = request.get_json()
        result = yield_service.predict(data)

        return jsonify({
            "yield_kg_per_ha": round(result, 2)
        })