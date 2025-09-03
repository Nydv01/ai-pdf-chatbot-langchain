import os
import uuid
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

# Import LangChain components
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --- Environment Setup ---
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not set in the environment.")

# --- Application Setup ---
app = FastAPI(title="AI PDF Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory Storage ---
# Simple dictionaries to hold session data in memory.
# In a production app, you'd use a database like Redis or a persistent store.
vector_stores = {}
chat_histories = {}

# --- Pydantic Models for API ---
class ProcessResponse(BaseModel):
    session_id: str
    message: str

class ChatRequest(BaseModel):
    session_id: str
    question: str

class ChatResponse(BaseModel):
    answer: str

# --- Core Logic Functions ---
def get_pdf_text(pdf_files: List[UploadFile]) -> str:
    """Extracts text from a list of uploaded PDF files."""
    text = ""
    for pdf_file in pdf_files:
        try:
            pdf_reader = PdfReader(pdf_file.file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            # Handle potential errors with corrupted PDFs
            print(f"Error reading {pdf_file.filename}: {e}")
            continue
    return text

def get_text_chunks(text: str) -> List[str]:
    """Splits a long text into manageable chunks."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vector_store(text_chunks: List[str]):
    """Creates a FAISS vector store from text chunks."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        return FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    except Exception as e:
        # This can fail if the API key is invalid or there are network issues.
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {e}")

# --- API Endpoints ---
@app.post("/process-pdfs/", response_model=ProcessResponse)
async def process_pdfs_endpoint(pdf_files: List[UploadFile]):
    if not pdf_files:
        raise HTTPException(status_code=400, detail="No PDF files uploaded.")

    # 1. Extract text from PDFs
    raw_text = get_pdf_text(pdf_files)
    if not raw_text:
        raise HTTPException(status_code=400, detail="Could not extract text from the provided PDF(s).")

    # 2. Split text into chunks
    text_chunks = get_text_chunks(raw_text)

    # 3. Create vector store
    vector_store = get_vector_store(text_chunks)

    # 4. Create a unique session ID and store the vector store
    session_id = str(uuid.uuid4())
    vector_stores[session_id] = vector_store
    
    # Initialize chat history for the new session
    chat_histories[session_id] = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )

    return {"session_id": session_id, "message": "PDFs processed successfully."}

@app.post("/chat/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    session_id = request.session_id
    question = request.question

    if session_id not in vector_stores:
        raise HTTPException(status_code=404, detail="Invalid session ID.")

    vector_store = vector_stores[session_id]
    memory = chat_histories.get(session_id)

    if not memory:
        raise HTTPException(status_code=500, detail="Chat history not found for session.")

    try:
        # Create the conversation chain
        llm = ChatOpenAI(openai_api_key=api_key)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory
        )

        # Get the answer using the modern .invoke() method
        response = conversation_chain.invoke({'question': question})
        answer = response['answer']
        
        return {"answer": answer}

    except Exception as e:
        # This line adds detailed logging to the terminal to find the root cause.
        print(f"An error occurred in the /chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {e}")

@app.get("/")
def read_root():
    return {"status": "API is running"}

