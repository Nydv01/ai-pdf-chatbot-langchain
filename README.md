AI PDF Chatbot with LangChain
An AI-powered chatbot that allows you to upload PDF documents and have a conversation with them. Ask questions, get summaries, and find information instantly. This project is built with a powerful FastAPI backend and a sleek, responsive React frontend.

Features
Multiple PDF Uploads: Upload one or more PDF documents at a time.

Secure Sessions: Each document processing session is unique and secure.

Conversational Memory: The chatbot remembers the context of your conversation within a session.

Real-time Interaction: Get answers from your documents instantly.

Clean & Modern UI: A user-friendly interface built with React and Tailwind CSS.

Tech Stack
Backend:

Frontend:

How to Run This Project Locally
Prerequisites
Python 3.10+

Node.js and npm (for potential future frontend development)

An OpenAI API Key

1. Clone the repository
git clone [https://github.com/Nydv01/ai-pdf-chatbot-langchain.git](https://github.com/Nydv01/ai-pdf-chatbot-langchain.git)
cd ai-pdf-chatbot-langchain

2. Setup the Backend
# Navigate to the backend directory
cd backend

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create a .env file and add your OpenAI API key
echo 'OPENAI_API_KEY="YOUR_API_KEY_HERE"' > .env

# Run the server
uvicorn app:app --reload

3. Launch the Frontend
Navigate to the frontend directory and open the index.html file in your browser. The application should now be connected to the backend server running on http://127.0.0.1:8000.
