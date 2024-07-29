# Penny App Backend
The backend chatbot for the penny app using FAISS and FastAPI

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Contributing](#contributing)
- [License](#license)

## Features

- Personalized responses to financial queries
- Context-aware information retrieval using FAISS
- Integration with Google's Gemini AI for natural language processing
- Conversation history tracking for each user
- FastAPI backend for efficient API handling

## Technologies Used

- Python 3.8+
- FastAPI
- Google Generative AI (Gemini)
- Sentence Transformers
- FAISS (Facebook AI Similarity Search)
- Pydantic
- Uvicorn

## Installation

Clone the repository:
bash
git clone https://github.com/yourusername/penny-app-backend.git
cd penny-app-backend
Create a virtual environment:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Set up your environment variables (see Environment Variables section).'''

## Usage
Start the FastAPI server:

bash
Copy code
uvicorn main:app --reload
The API will be available at http://127.0.0.1:8000.
Use the /chat endpoint to interact with the chatbot.

API Endpoints
POST /chat
Sends a message to the chatbot and receives a response.

Request body:

json
Copy code
{
    "user_id": "string",
    "message": "string"
}
Response body:

json
Copy code
{
    "response": "string",
    "history": ["string"]
}
Environment Variables
Create a .env file in the root directory with the following variables:

dotenv
Copy code
GEMINI_API_KEY=your_gemini_api_key_here
Make sure to replace your_gemini_api_key_here with your actual Gemini API key.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
