import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


class FarmAdvisorChat:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-1.5-flash-latest")
        self.sessions = {}   # stores chats by session id

    def start_crop_chat(self, session_id, crop_result, input_data):
        """
        Starts a chat after crop prediction
        """

        prompt = f"""
You are an agricultural expert assistant.

The machine learning model recommended crop: {crop_result['crop']}
Confidence: {crop_result['confidence']}

User soil/environment data:
Nitrogen: {input_data['N']}
Phosphorus: {input_data['P']}
Potassium: {input_data['K']}
Temperature: {input_data['temperature']}
Humidity: {input_data['humidity']}
pH: {input_data['ph']}
Rainfall: {input_data['rainfall']}

Explain WHY this crop was recommended in simple farmer-friendly language.
Mention possible advantages.
Be concise but helpful.
"""

        chat = self.model.start_chat(history=[])

        response = chat.send_message(prompt)

        self.sessions[session_id] = chat

        return response.text


    def start_yield_chat(self, session_id, predicted_yield, input_data):
        prompt = f"""
Predicted yield:
{predicted_yield} kg/ha

Inputs:
State: {input_data['state_name']}
Crop: {input_data['crop']}
Area: {input_data['area_ha']}
Temperature: {input_data['temperature']}
Humidity: {input_data['humidity']}
pH: {input_data['ph']}
Rainfall: {input_data['rainfall']}

Explain why this yield may occur and how it can be improved.
"""

        chat = self.model.start_chat(history=[])

        response = chat.send_message(prompt)

        self.sessions[session_id] = chat

        return response.text


    def continue_chat(self, session_id, user_message):

        if session_id not in self.sessions:
            return "No active chat found."

        response = self.sessions[session_id].send_message(
            user_message
        )

        return response.text