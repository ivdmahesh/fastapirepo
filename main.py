from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

class Message(BaseModel):
    message: str

@app.post("/chat", description="chat with rag application through this endpoint")
def chat(message:Message):
    try:
        response = get_answer_and_docs(message.message)
        response_content = {
        "question":message.message,
        "answer":response["answer"],
        "documents":[doc for doc in response["context"]]
        }
        return JSONResponse(content=response_content, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message":"Exception occured: "+str(e)}, status_code=404)

def get_answer_and_docs(question:str):
    return {
        "answer":f"answer to the question: {question}",
        "context":[{"contextkey":"context will come here"}]
    }