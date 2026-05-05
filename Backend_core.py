from flask import Flask, request, jsonify, Response, stream_with_context
from supabase import create_client
from flask_cors import CORS
from dotenv import load_dotenv
import ml_service
from ml_service import register_ml_routes
from ai_chat_service import FarmAdvisorChat
import uuid
import os
import jwt
import datetime
import bcrypt
from functools import wraps

#LOAD ENV VARIABLES -------------------- #
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")

if not SUPABASE_URL or not SUPABASE_KEY or not JWT_SECRET:
    raise ValueError("Missing environment variables in Render dashboard")

#APP SETUP -------------------- #
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
app = Flask(__name__)

CORS(app, origins=["*"])
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# Initialize services
chat_service = FarmAdvisorChat()

#OPTIONAL PROTECTED ROUTE DECORATOR -------------------- #
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None

        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]

        if not token:
            return jsonify({"message": "Token missing"}), 401

        try:
            decoded = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            request.user_id = decoded["user_id"]
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401

        return f(*args, **kwargs)

    return decorated


# -------------------- REGISTER -------------------- #
@app.route('/register', methods=['POST'])
def register():
    try:
        data = request.get_json()

        required_fields = [
            "name",
            "phone_email",
            "password",
            "location",
            "farm_size",
            "primary_crop"
        ]

        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({
                    "success": False,
                    "message": f"Missing field: {field}"
                }), 400
                
        # Check existing user
        existing_user = (
            supabase
            .table("users")
            .select("*")
            .eq("phone_email", data["phone_email"])
            .execute()
        )

        if existing_user.data:
            return jsonify({
                "success": False,
                "message": "User already exists"
            }), 409

        # Hash password
        hashed_password = bcrypt.hashpw(
            data["password"].encode('utf-8'),
            bcrypt.gensalt()
        ).decode('utf-8')

        # Insert user
        supabase.table("users").insert({
            "name": data["name"],
            "phone_email": data["phone_email"],
            "password": hashed_password,
            "location": data["location"],
            "farm_size": data["farm_size"],
            "primary_crop": data["primary_crop"]
        }).execute()

        return jsonify({
            "success": True,
            "message": "Registered successfully"
        }), 201

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# -------------------- CHECK USER EXISTS -------------------- #
@app.route('/check-user', methods=['POST'])
def check_user():
    try:
        data = request.get_json()

        if "phone_email" not in data or not data["phone_email"]:
            return jsonify({"message": "phone_email is required"}), 400

        user = (
            supabase
            .table("users")
            .select("user_id")
            .eq("phone_email", data["phone_email"])
            .execute()
        )

        if user.data:
            return jsonify({
                "exists": True,
                "message": "User found"
            }), 200
        else:
            return jsonify({
                "exists": False,
                "message": "User not found"
            }), 404

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# -------------------- LOGIN -------------------- #
@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.get_json()

        if "phone_email" not in data or "password" not in data:
            return jsonify({"message": "Email/phone and password required"}), 400

        user = (
            supabase
            .table("users")
            .select("*")
            .eq("phone_email", data["phone_email"])
            .execute()
        )

        if not user.data:
            return jsonify({
                "redirect": "signup",
                "message": "User not found"
            }), 404

        db_user = user.data[0]

        # Verify password
        if not bcrypt.checkpw(
            data["password"].encode('utf-8'),
            db_user["password"].encode('utf-8')
        ):
            return jsonify({"message": "Wrong password"}), 401

        # Create JWT token
        token = jwt.encode(
            {
                "user_id": db_user["user_id"],
                "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=15)
            },
            JWT_SECRET,
            algorithm="HS256"
        )

        return jsonify({
            "success": True,
            "token": token
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -------------------- CROP PREDICTION ROUTE -------------------- #
@app.route('/predict_with_chat', methods=['POST'])
def predict_crop_with_chat():
    try:
        data = request.get_json()

        if ml_service.crop_service is None:
            return jsonify({
                "success": False,
                "error": "Crop prediction service not available. Please try again later."
            }), 503

        result = ml_service.crop_service.predict(data)

        session_id = str(uuid.uuid4())

        explanation = chat_service.start_crop_chat(
            session_id,
            result,
            data
        )

        return jsonify({
            "success": True,
            "crop": result["crop"],
            "confidence": result["confidence"],
            "explanation": explanation,
            "chat_session": session_id
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# -------------------- YIELD PREDICTION ROUTE -------------------- #
@app.route('/predict_yield_with_chat', methods=['POST'])
def predict_yield_with_chat():
    try:
        data = request.get_json()
        
        if ml_service.yield_service is None:
            return jsonify({
                "success": False,
                "error": "Yield prediction service not available. Please try again later."
            }), 503
        
        result = ml_service.yield_service.predict(data)
        
        session_id = str(uuid.uuid4())
        
        explanation = chat_service.start_yield_chat(
            session_id,
            result,
            data
        )
        
        return jsonify({
            "success": True,
            "yield_kg_per_ha": round(result, 2),
            "explanation": explanation,
            "chat_session": session_id
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -------------------- CHAT ROUTE -------------------- #
@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        
        if "message" not in data:
            return jsonify({
                "success": False,
                "error": "message is required"
            }), 400

        chat_session = data.get("chat_session") or str(uuid.uuid4())
        
        @stream_with_context
        def generate():
            # Force an immediate flush to bypass browser/proxy buffering
            yield "data: \n\n" 
            for chunk in chat_service.continue_chat_stream(chat_session, data["message"]):
                if chunk:
                    # SSE format: "data: <content>\n\n"
                    yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'X-Accel-Buffering': 'no',
                'Cache-Control': 'no-cache, no-transform',
                'Connection': 'keep-alive',
                'X-Chat-Session-ID': chat_session
            }
        )
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# -------------------- TEST PROTECTED ROUTE -------------------- #
@app.route('/profile', methods=['GET'])
@token_required
def profile():
    return jsonify({
        "message": "Protected route accessed",
        "user_id": request.user_id
    })


# -------------------- HEALTH CHECK -------------------- #
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy"
    })


# -------------------- REGISTER ML ROUTES -------------------- #
print("Before ML routes")
register_ml_routes(app)
print("After ML routes")

# -------------------- RUN -------------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)