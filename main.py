import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from models import ChatRequest, ChatResponse
from catalog import load_catalog
from agent import call_agent, init_client

# Load .env before anything else
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalog and init Groq client at startup."""
    load_catalog()
    init_client()
    yield


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational AI agent for recommending SHL assessments",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend or external callers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    Main chat endpoint. Accepts conversation history, returns reply,
    recommendations, and end_of_conversation flag.
    """
    try:
        reply, recommendations, end_of_conversation = call_agent(request.messages)
        return ChatResponse(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
