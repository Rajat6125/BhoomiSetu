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

        # Map of expected feature names to actual column names in Supabase
        self.feature_mapping = {
            'n': 'N',           # Note: uppercase N in your table
            'p': 'P',           # uppercase P
            'k': 'K',           # uppercase K
            'temperature': 'temperature',
            'humidity': 'humidity',
            'ph': 'ph',
            'rainfall': 'rainfall'
        }
        
        self.features = list(self.feature_mapping.keys())

    def load_data(self):
        res = supabase.table("crop_recommendation").select("*").execute()
        df = pd.DataFrame(res.data)

        if df.empty:
            raise ValueError("crop_recommendation table is empty")

        print("Original crop columns:", df.columns.tolist())
        
        # No need to rename columns, just work with original names
        # The label column is fine as 'label'
        
        if 'label' not in df.columns:
            raise ValueError("Column 'label' not found in crop_recommendation table")

        crops = sorted(df['label'].unique())
        self.id_map = {i: c for i, c in enumerate(crops)}
        reverse = {c: i for i, c in self.id_map.items()}

        df['label_id'] = df['label'].map(reverse)

        return df

    def train(self):
        df = self.load_data()

        # Use actual column names from Supabase
        X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
        y = df['label_id']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            eval_metric="mlogloss",
            random_state=42
        )

        self.model.fit(X_train, y_train)
        print(f"✅ Crop model trained. Test accuracy: {self.model.score(X_test, y_test):.2f}")

    def predict(self, data):
        # Map incoming data (which uses lowercase keys) to uppercase column names
        x = np.array([[
            float(data['N']),           # Note: expecting uppercase N in request
            float(data['P']),           # uppercase P
            float(data['K']),           # uppercase K
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

        print("Original yield columns:", df.columns.tolist())
        
        # Check for column names with spaces - they need special handling
        # In your table, 'Dist Code' has a space, 'State Name' has space, etc.
        
        # Rename columns with spaces to avoid issues
        df.columns = df.columns.str.replace(' ', '_', regex=False)
        
        print("Renamed yield columns:", df.columns.tolist())

        return df

    def train(self):
        df = self.load_data()

        # Create mappings for state and crop
        self.state_map = {
            s: i for i, s in enumerate(df['State_Name'].unique())
        }

        self.crop_map = {
            c: i for i, c in enumerate(df['Crop'].unique())
        }

        df['state_code'] = df['State_Name'].map(self.state_map)
        df['crop_code'] = df['Crop'].map(self.crop_map)

        # Use actual column names from your table
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
        print(f"✅ Yield model trained. R² score: {self.model.score(X_test, y_test):.2f}")

    def predict(self, data):
        # Map incoming data to column names used in training
        x = [[
            self.state_map[data['state_name']],
            self.crop_map[data['crop']],
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


# ---------------- CREATE GLOBAL INSTANCES ---------------- #
# Create the service instances at module level so they can be imported
crop_service = None
yield_service = None

# ---------------- REGISTER ROUTES ---------------- #
def register_ml_routes(app):
    global crop_service, yield_service
    
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