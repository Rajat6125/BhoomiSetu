import os
from openai import OpenAI

# OpenRouter client
client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


class FarmAdvisorChat:

    def __init__(self):
        # stores full message history for each user session
        self.sessions = {}

        # choose a model from OpenRouter
        self.model = "meta-llama/llama-3.1-8b-instruct"

    def ask_model(self, session_id, user_prompt):

        # Initialize session with STRICT system prompt
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert agricultural advisor helping farmers. "
                        "Always reply in SHORT, DIRECT points. "
                        "Max 2-4 lines per response. "
                        "No long explanations, no extra text, no exaggeration. "
                        "Use simple farmer-friendly language. "
                        "If answer is long, split into 2-4 parts using '||'."
                    )
                }
            ]

        # add user message
        self.sessions[session_id].append(
            {
                "role": "user",
                "content": user_prompt
            }
        )

        response = client.chat.completions.create(
            model=self.model,
            messages = self.sessions[session_id][-6:]  ,
            temperature=0.5,
            max_tokens=60   # limits response size
        )

        reply = response.choices[0].message.content.strip()

        # Split into multiple small messages if needed
        parts = [p.strip() for p in reply.split("||") if p.strip()]

        # Save each part separately
        for part in parts:
            self.sessions[session_id].append(
                {
                    "role": "assistant",
                    "content": part
                }
            )

        return parts  # returns list of short messages


    def start_crop_chat(self, session_id, crop_result, input_data):

        prompt = f"""
Crop: {crop_result['crop']}
Confidence: {crop_result['confidence']}

Conditions:
N={input_data['N']}, P={input_data['P']}, K={input_data['K']}
Temp={input_data['temperature']}, Humidity={input_data['humidity']}
pH={input_data['ph']}, Rainfall={input_data['rainfall']}

Give:
- Why this crop (1 line)
- Suitability (1 line)
- Benefit (1 line)

Keep answer VERY SHORT.
"""

        return self.ask_model(session_id, prompt)


    def start_yield_chat(self, session_id, predicted_yield, input_data):

        prompt = f"""
Yield: {predicted_yield} kg/ha

Crop: {input_data['crop']}
Area: {input_data['area_ha']}
Temp: {input_data['temperature']}
Humidity: {input_data['humidity']}
pH: {input_data['ph']}
Rainfall: {input_data['rainfall']}

Give:
- Reason for this yield (1-2 lines)
- 1 improvement tip

Keep it SHORT.
"""

        return self.ask_model(session_id, prompt)


    def continue_chat(self, session_id, user_message):

        if session_id not in self.sessions:
            return ["No active chat found."]

        return self.ask_model(session_id, user_message)