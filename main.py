from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel('gemini-1.5-pro')

# In-memory store for conversation history
conversation_history: Dict[str, List[str]] = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    history: List[str]

def get_history(user_id: str) -> List[str]:
    return conversation_history.get(user_id, [])

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Generate a response
        response = model.generate_content(request.message)
        response_text = response.text

        # Store the conversation history
        if request.user_id not in conversation_history:
            conversation_history[request.user_id] = []
        conversation_history[request.user_id].append(f"User: {request.message}")
        conversation_history[request.user_id].append(f"Bot: {response_text}")

        # Return the response and history
        history = get_history(request.user_id)
        return ChatResponse(response=response_text, history=history)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
