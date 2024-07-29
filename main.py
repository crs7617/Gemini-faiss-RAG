import logging
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn
from typing import List, Dict
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import fitz  # PyMuPDF

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
if not os.path.exists('.env'):
    logger.warning("No .env file found. Make sure to set the GEMINI_API_KEY environment variable.")
load_dotenv()

app = FastAPI()

# Constants
MODEL_NAME = 'gemini-1.5-pro'
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 3

# Configure Gemini API
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL_NAME)

# In-memory store for conversation history
conversation_history: Dict[str, List[str]] = {}

# Load sentence transformer model
sentence_model = SentenceTransformer(SENTENCE_MODEL_NAME)

# Prepare finance information
finance_info: List[str] = [
    "Personal finance refers to managing one's financial resources effectively, including earning, spending, saving, investing, and protecting against risks.",
    # ... (rest of the finance_info list)
]

# Create FAISS index
finance_embeddings = sentence_model.encode(finance_info)
dimension = finance_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(finance_embeddings.astype('float32'))

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    history: List[str]

def get_history(user_id: str) -> List[str]:
    return conversation_history.get(user_id, [])

def get_relevant_context(query: str, top_k: int = TOP_K) -> str:
    query_vector = sentence_model.encode([query])
    distances, indices = index.search(query_vector.astype('float32'), top_k)
    relevant_points = [finance_info[i] for i in indices[0]]
    return " ".join(relevant_points)

def extract_text_from_pdf(file: UploadFile) -> str:
    document = fitz.open(stream=file.file.read(), filetype="pdf")
    text = ""
    for page in document:
        text += page.get_text()
    return text

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        # Get relevant context from finance information
        context = get_relevant_context(request.message)
        
        # Prepare the prompt with context
        prompt = f"Based on the following context from our finance course: '{context}', please answer the user's question: {request.message}"
        
        # Generate a response
        response = model.generate_content(prompt)
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
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile):
    try:
        text = extract_text_from_pdf(file)
        finance_info.append(text)
        
        # Update FAISS index with the new text
        new_embeddings = sentence_model.encode([text])
        index.add(new_embeddings.astype('float32'))
        
        return {"message": "PDF content added successfully."}

    except Exception as e:
        logger.error(f"An error occurred while processing the PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
