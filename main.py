from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI()

class Message(BaseModel):
    message: str

@app.post("/chat", description="chat with rag application through this endpoint")
def chat(message:Message):
    try:
        return JSONResponse(content={"message":message.message}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message":"Exception occured: "+str(e)}, status_code=404)
