"""
Backend API for an AI PDF Chatbot using FastAPI and LangChain.

This module provides endpoints to upload and process PDF files, create a
vector store for their content, and interact with the documents through a
conversational chat interface.
"""

# --- Standard Library Imports ---
import os
import uuid
from typing import List, Dict, Any

# --- Third-party Library Imports ---
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

# --- LangChain Imports ---
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- Environment Setup ---
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")

# --- Application Setup ---
app = FastAPI(
    title="AI PDF Chatbot API",
    description="An API to chat with your PDF documents.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory Storage ---
# Note: In a production app, replace these with a persistent database like Redis.
vector_stores: Dict[str, FAISS] = {}
chat_histories: Dict[str, ConversationBufferMemory] = {}


# --- Pydantic Models for API Data Structure ---
class ProcessResponse(BaseModel):
    """Response model for the PDF processing endpoint."""
    session_id: str
    message: str


class ChatRequest(BaseModel):
    """Request model for the chat endpoint."""
    session_id: str
    question: str


class ChatResponse(BaseModel):
    """Response model for the chat endpoint."""
    answer: str


# --- Core Logic Functions ---
def get_pdf_text(pdf_files: List[UploadFile]) -> str:
    """Extracts and concatenates text from a list of uploaded PDF files."""
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_file.file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except PdfReadError:
            print(f"Warning: Could not read corrupted PDF: {pdf_file.filename}")
            continue
        except (AttributeError, TypeError, ValueError) as e:
            print(f"An unexpected error occurred with {pdf_file.filename}: {e}")
            continue
    return text


def get_text_chunks(text: str) -> List[str]:
    """Splits a long text into manageable chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)


def get_vector_store(text_chunks: List[str]) -> FAISS:
    """Creates a FAISS vector store from text chunks using OpenAI embeddings."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        # This exception is intentionally broad to catch any potential OpenAI API
        # or network errors and report them cleanly to the user.
        raise HTTPException(
            status_code=500, detail=f"Failed to create vector store: {str(e)}"
        ) from e


# --- API Endpoints ---
@app.post("/process-pdfs/", response_model=ProcessResponse)
async def process_pdfs_endpoint(pdf_files: List[UploadFile]):
    """
    Processes uploaded PDF files, creates a vector store, and returns a session ID.
    """
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")

    raw_text = get_pdf_text(pdf_files)
    if not raw_text:
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from the provided PDF(s)."
        )

    text_chunks = get_text_chunks(raw_text)
    vector_store = get_vector_store(text_chunks)

    # Create and store session data for the new conversation
    session_id = str(uuid.uuid4())
    vector_stores[session_id] = vector_store
    chat_histories[session_id] = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    return {"session_id": session_id, "message": "PDFs processed successfully."}


@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Handles a user's question and returns the AI-generated answer."""
    session_id = request.session_id
    question = request.question

    vector_store = vector_stores.get(session_id)
    memory = chat_histories.get(session_id)

    if not vector_store or not memory:
        raise HTTPException(status_code=404, detail="Invalid session ID.")

    try:
        llm = ChatOpenAI(openai_api_key=API_KEY, temperature=0.7)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vector_store.as_retriever(), memory=memory
        )

        response: Dict[str, Any] = conversation_chain.invoke({'question': question})
        answer = response.get('answer', "Sorry, I couldn't find an answer.")

        return {"answer": answer}

    except Exception as e:
        # This exception is intentionally broad to catch any LangChain or API
        # errors during the conversation and report them cleanly.
        print(f"An error occurred in the /chat endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(e)}"
        ) from e


@app.get("/")
def read_root() -> Dict[str, str]:
    """Root endpoint to confirm the API is running."""
    return {"status": "API is running"}
