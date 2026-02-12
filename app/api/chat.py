from fastapi import APIRouter
from pydantic import BaseModel
from app.agent.rag import chat_with_knowledge

router = APIRouter()

class ChatRequest(BaseModel):
    question: str


@router.post("/chat")
async def chat(req: ChatRequest):
    answer = await chat_with_knowledge(req.question)  # ✅ FIXED
    return {"answer": answer}
