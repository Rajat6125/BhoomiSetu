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
        # other options:
        # mistralai/mistral-7b-instruct
        # google/gemini-2.0-flash-exp
        # openai/gpt-4o-mini


    def ask_model(self, session_id, user_prompt):

        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {
                    "role":"system",
                    "content":"You are an expert agricultural advisor helping farmers."
                }
            ]

        # add user message
        self.sessions[session_id].append(
            {
                "role":"user",
                "content":user_prompt
            }
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=self.sessions[session_id],
            temperature=0.7
        )

        reply = response.choices[0].message.content

        # save assistant reply in chat memory
        self.sessions[session_id].append(
            {
                "role":"assistant",
                "content":reply
            }
        )

        return reply


    def start_crop_chat(self, session_id, crop_result, input_data):

        prompt = f"""
The ML model recommended crop: {crop_result['crop']}
Confidence: {crop_result['confidence']}

Inputs:
Nitrogen: {input_data['N']}
Phosphorus: {input_data['P']}
Potassium: {input_data['K']}
Temperature: {input_data['temperature']}
Humidity: {input_data['humidity']}
pH: {input_data['ph']}
Rainfall: {input_data['rainfall']}

Explain:
1. Why this crop was recommended
2. Why these conditions suit it
3. Benefits of growing it
Use simple farmer-friendly language.
"""

        return self.ask_model(session_id, prompt)


    def start_yield_chat(self, session_id, predicted_yield, input_data):

        prompt = f"""
Predicted Yield: {predicted_yield} kg/ha

Inputs:
State: {input_data['state_name']}
Crop: {input_data['crop']}
Area: {input_data['area_ha']}
Temperature: {input_data['temperature']}
Humidity: {input_data['humidity']}
pH: {input_data['ph']}
Rainfall: {input_data['rainfall']}

Explain why this yield may occur and how farmer can improve it.
"""

        return self.ask_model(session_id, prompt)


    def continue_chat(self, session_id, user_message):

        if session_id not in self.sessions:
            return "No active chat found."

        return self.ask_model(session_id, user_message)