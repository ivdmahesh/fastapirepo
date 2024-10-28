from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from decouple import config
import os

from langchain_openai import  ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient, models
from langchain_qdrant.qdrant import QdrantVectorStore

from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://frontend-1tmk1uomi-maheshs-projects-6dabae76.vercel.app/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"]
)

prod = True



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

retriever = vector_store.as_retriever()

prompt_template = """
Answer the question based on the context, in a concise manner, in markdown and using bullet points, giving importance to timelines and locations where applicable.

Context: {context}
Question: {question}
Answer:
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

def create_chain():
    chain = (
        {
            "context":retriever.with_config(top_k=4),
            "question":RunnablePassthrough()
        }
        | RunnableParallel({
            "response": prompt | model,
            "context": itemgetter("context")
        })
    )
    return chain


class Message(BaseModel):
    message: str

@app.post("/chat", description="chat with rag application through this endpoint")
def chat(message:Message):
    try:
        response = get_answer_and_docs(message.message)
        response_content = {
        "question":message.message,
        "answer":response["answer"],
        "documents":[doc.dict() for doc in response["context"]]
        }
        return JSONResponse(content=response_content, status_code=200)
    except Exception as e:
        return JSONResponse(content={"message":"Exception occured: "+str(e)}, status_code=404)

def get_answer_and_docs(question:str):
    chain=create_chain()
    response = chain.invoke(question)
    answer = response["response"].content
    context = response["context"]
    return {
        "answer":answer,
        "context":context
    }