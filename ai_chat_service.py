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

        # Initialize session with IMPROVED system prompt
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful agricultural advisor for farmers.\n\n"
                        
                        "RESPONSE GUIDELINES:\n"
                        "• Keep answers SHORT and MEANINGFUL: 2-3 sentences max for direct questions\n"
                        "• Be DIRECT: Answer immediately, no fluff or extra explanation\n"
                        "• Use SIMPLE language: Explain like talking to a farmer, not textbooks\n"
                        "• Include DATA when relevant: Numbers, measurements, percentages, specific recommendations\n"
                        "• Use emojis sparingly (max 1-2 per response) to add warmth only\n"
                        "• For long answers, split with ' || ' (max 3 parts)\n\n"
                        
                        "RESPONSE STRUCTURE:\n"
                        "• 'WHAT/HOW' questions → Answer in points with specific data/steps\n"
                        "• 'WHY' questions → Brief explanation (50-80 words) with reasons\n"
                        "• 'PROBLEM' questions → Diagnosis + 2-3 actionable solutions\n\n"
                        
                        "FORMATTING RULES (IMPORTANT):\n"
                        "• NO markdown symbols: Do NOT use **text**, __text__, *text*, ~~text~~\n"
                        "• Send plain text only\n"
                        "• Use CAPS for emphasis if needed: IMPORTANT TEXT\n"
                        "• Use dash for sub-points: - Point 1, - Point 2\n"
                        "• Numbers clearly: 5kg, 25°C, 6.5pH (not five kilograms)\n"
                        "• Split long answers with ' || ' separator\n\n"
                        
                        "QUALITY CHECKLIST:\n"
                        "✓ Actionable? Can the farmer do something with this?\n"
                        "✓ Accurate? Use real agricultural knowledge\n"
                        "✓ Concise? No unnecessary details\n"
                        "✓ Relevant? Directly answers the question\n"
                        "✓ Encouraging? Helpful tone, not robotic\n\n"
                        
                        "AVOID:\n"
                        "✗ Long paragraphs or essays\n"
                        "✗ Markdown formatting\n"
                        "✗ Excessive emojis or exclamation marks\n"
                        "✗ Vague or generic answers\n"
                        "✗ Off-topic information"
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
            messages=self.sessions[session_id][-6:],  # keep last 6 messages for context
            temperature=0.5,
            max_tokens=150  # increased from 60 to allow for quality responses
        )

        reply = response.choices[0].message.content.strip()

        # Clean up any markdown symbols that might slip through
        reply = self._clean_markdown(reply)

        # Split into multiple small messages if needed (by || separator)
        parts = [p.strip() for p in reply.split("||") if p.strip()]

        # If no || separator, keep as single response
        if not parts:
            parts = [reply]

        # Save the full response to session history as one message
        # This preserves context for follow-up questions
        self.sessions[session_id].append(
            {
                "role": "assistant",
                "content": reply
            }
        )

        return parts  # returns list of short messages


    def ask_model_stream(self, session_id, user_prompt):
        # Initialize session with system prompt if needed
        if session_id not in self.sessions:
            self.sessions[session_id] = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful agricultural advisor for farmers.\n\n"
                        "RESPONSE GUIDELINES:\n"
                        "• Keep answers SHORT and MEANINGFUL: 2-3 sentences max for direct questions\n"
                        "• Be DIRECT: Answer immediately, no fluff or extra explanation\n"
                        "• Use SIMPLE language: Explain like talking to a farmer, not textbooks\n"
                        "• Include DATA when relevant: Numbers, measurements, percentages, specific recommendations\n"
                        "• Use emojis sparingly (max 1-2 per response) to add warmth only\n"
                        "• FORMATTING: Use plain text only. NO markdown symbols like **text** or __text__.\n"
                    )
                }
            ]

        # Add user message
        self.sessions[session_id].append({"role": "user", "content": user_prompt})

        # Create stream
        response = client.chat.completions.create(
            model=self.model,
            messages=self.sessions[session_id][-6:],
            temperature=0.5,
            max_tokens=250,
            stream=True
        )

        full_reply = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                full_reply += content
                yield content

        # Save the full response to session history
        self.sessions[session_id].append({"role": "assistant", "content": full_reply})

    def continue_chat_stream(self, session_id, user_message):
        """
        Continue existing chat session with follow-up questions (streaming)
        """
        if session_id not in self.sessions:
            # Fallback to initializing a new session if not found
            return self.ask_model_stream(session_id, user_message)

        return self.ask_model_stream(session_id, user_message)


    def _clean_markdown(self, text):
        """Remove markdown symbols that might interfere with display"""
        # Remove bold (**text**)
        text = text.replace('**', '')
        # Remove italic (*text* or _text_)
        text = text.replace('*', '')
        text = text.replace('_', '')
        # Remove strikethrough (~~text~~)
        text = text.replace('~~', '')
        return text


    def start_crop_chat(self, session_id, crop_result, input_data):
        """
        Analyze recommended crop with conditions
        """
        prompt = f"""
CROP RECOMMENDATION:
Crop: {crop_result['crop']}
Confidence: {crop_result['confidence']}%

SOIL & ENVIRONMENTAL CONDITIONS:
NPK: N={input_data['N']}, P={input_data['P']}, K={input_data['K']}
Temperature: {input_data['temperature']}°C
Humidity: {input_data['humidity']}%
Soil pH: {input_data['ph']}
Rainfall: {input_data['rainfall']}mm

ANSWER BRIEFLY (plain text, no bold):
- Why suitable for this crop? (1 line with reason)
- Suitability level? (1 line - good/excellent/moderate)
- Main benefit? (1 line - yield/profit/soil health)

Keep it SHORT and MEANINGFUL. Use plain text only.
"""
        # Note: Keeping this non-streaming for now as it's used in special forms
        # but we could stream it too if needed.
        return self.ask_model(session_id, prompt)


    def start_yield_chat(self, session_id, predicted_yield, input_data):
        """
        Explain predicted yield and give improvement tips
        """
        prompt = f"""
YIELD PREDICTION:
Expected Yield: {predicted_yield} kg/ha

CROP & CONDITIONS:
Crop: {input_data['crop']}
Area: {input_data['area_ha']} hectares
Temperature: {input_data['temperature']}°C
Humidity: {input_data['humidity']}%
Soil pH: {input_data['ph']}
Rainfall: {input_data['rainfall']}mm

ANSWER (plain text only, no markdown):
- Why this yield? (1-2 lines - what factors affect it)
- Top improvement tip (1 line - specific action)
- Expected increase (1 line - if tip applied, how much better)

Be specific with numbers and actionable. Keep SHORT.
"""
        return self.ask_model(session_id, prompt)


    def continue_chat(self, session_id, user_message):
        """
        Continue existing chat session with follow-up questions
        """
        if session_id not in self.sessions:
            return ["No active chat found. Start a new conversation."]

        return self.ask_model(session_id, user_message)


# ==================== USAGE EXAMPLE ====================

if __name__ == "__main__":
    chat = FarmAdvisorChat()
    