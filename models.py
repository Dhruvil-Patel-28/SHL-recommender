from pydantic import BaseModel
from typing import List


class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str  # e.g. "K" or "P" or "K,S"


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]  # empty list [] when not recommending, never null
    end_of_conversation: bool
