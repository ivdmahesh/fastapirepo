from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from decouple import config
import os

from langchain_openai import  ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from langchain_qdrant.qdrant import QdrantVectorStore

app = FastAPI()

prod = False

if(prod):
    qdrantendpoint = os.environ['qdrantendpoint']
    qdrantapikey = os.environ['qdrantapikey']
    openaiapikey = os.environ['openaiapikey']
else:
    qdrantendpoint = config("qdrantendpoint")
    qdrantapikey = config("qdrantapikey")
    openaiapikey = config("openaiapikey")

collection_name= "websites"

model = ChatOpenAI(

    model_name="gpt-3.5-turbo",
    openai_api_key=openaiapikey,
    temperature=0
)

client=QdrantClient(
    url=qdrantendpoint,
    api_key=qdrantapikey
)

existingcollections = client.get_collections().collections

existingcollection_names=[c.name for c in existingcollections]

new_collection = collection_name not in existingcollection_names

if(new_collection):
    client.create_collection(
              collection_name=collection_name,
              vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
         )

vector_store = QdrantVectorStore(client=client, collection_name= collection_name, 
                                 embedding=OpenAIEmbeddings(api_key=openaiapikey))

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