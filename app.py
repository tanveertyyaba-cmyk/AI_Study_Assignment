from fastapi import FastAPI
from pydantic import BaseModel

from rag import retrieve
from grok import ask_grok

app = FastAPI(title="AI Study Assistant")

class Question(BaseModel):
    question: str

@app.get("/")
def home():
    return {"message": "AI Study Assistant is running"}

@app.post("/ask")
def ask(question: Question):
    chunks = retrieve(question.question)

    context = "\n\n".join(chunks)

    answer = ask_grok(
        question.question,
        context
    )

    return {
        "question": question.question,
        "answer": answer,
        "sources": chunks
    }