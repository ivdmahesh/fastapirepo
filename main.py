from fastapi import FastAPI, Query
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

from qdrant_client.http.models import Filter, FieldCondition, MatchValue

from langchain_text_splitters import RecursiveCharacterTextSplitter

import pymongo
from bson.objectid import ObjectId
from datetime import datetime, timezone

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://frontend-9kba3honx-maheshs-projects-6dabae76.vercel.app",
    "https://frontend-livid-six-56.vercel.app",
    "https://www.inmycolony.com"
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
    mongodbclienturl = os.environ['mongodbclienturl']
else:
    qdrantendpoint = config("qdrantendpoint")
    qdrantapikey = config("qdrantapikey")
    openaiapikey = config("openaiapikey")
    mongodbclienturl = config("mongodbclienturl")


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

mongoclient = pymongo.MongoClient(mongodbclienturl)
db = mongoclient.MERNLoginGoogle


class Message(BaseModel):
    message: str

class MessageText(BaseModel):
    text: str
    id:str

class MessageId(BaseModel):
    id: str

@app.get("/commonquestions", description="get the answers for common questions")
def commonquestions(id: str = Query(...)):
    response = get_commonquestions(id)
    response_content = {
        "answers":response["answers"]
    }
    return JSONResponse(content=response_content, status_code=200)

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
    
@app.post("/storeprofile", description="store pdf in profile collection through this endpoint")
def storeprofile(text:MessageText):
    try:
        profile_content = {
        "text":text.text,
        "id":text.id}
        response = upload_profile_to_collection(profile_content)
        question1 = "Who are we?"
        question2 ="What are we looking for?"
        question3 = "How to get in touch with us?"
        response1=get_answer_and_docs_from_profile(question1)
        response2=get_answer_and_docs_from_profile(question2)
        response3=get_answer_and_docs_from_profile(question3)
        collection_questionanswer = [{question1:response1["answer"]},{question2:response2["answer"]},{question3:response3["answer"]}]
        storecommonquestionandanswers(text.id,collection_questionanswer)
        return JSONResponse(content={"response":response}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/profilechat", description="chat with rag application through this endpoint")
def chat(message:Message):
    try:
        response = get_answer_and_docs_from_profile(message.message)
        response_content = {
        "question":message.message,
        "answer":response["answer"]}
        return JSONResponse(content=response_content, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)

@app.post("/refreshQuestion", description="refresh answer through this endpoint")
def chat(id:MessageId):
    try:
        questionanswer_id = ObjectId(id.id)
        questionanswer_collection = db.questionanswers
        questionanswer = questionanswer_collection.find_one(filter={"_id": questionanswer_id})
        
        response = get_answer_and_docs_from_profile(questionanswer['question'])
        
        updates = {
            "$set": {
                "answer":response['answer'],
                "updatedAt":datetime.now(timezone.utc)
                }}
        questionanswer_collection.update_one( filter={"_id": questionanswer_id}, update=updates)
        return JSONResponse(content={"response":"update successful"}, status_code=200)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


def get_answer_and_docs(question:str):
    chain=create_chain()
    response = chain.invoke(question)
    answer = response["response"].content
    context = response["context"]
    return {
        "answer":answer,
        "context":context
    }

def get_commonquestions(id):
    commonquestions_collection = db.commonquestions
    record=commonquestions_collection.find_one({"id":id})
    return record

def upload_profile_to_collection(profile_content):
    collection_name = "profile"
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=20,
                                                   length_function=len)
    docs = text_splitter.create_documents([profile_content["text"]])
    for doc in docs:
        doc.metadata = {"id":profile_content["id"]}
    
    client.delete(
    collection_name=collection_name,
    points_selector=models.FilterSelector(
        filter=models.Filter(must=[
        models.FieldCondition(
            key="metadata.id",
            match=models.MatchValue(value=profile_content["id"])
        )
    ])  # Empty filter matches all points
    ))

    vector_store.add_documents(docs)

def get_answer_and_docs_from_profile(question:str):
    chain=create_chain_profile()
    response = chain.invoke(question)
    answer = response["response"].content
    context = response["context"]
    return {
        "answer":answer,
        "context":context
    }

def create_chain_profile():
    collection_name = "profile"
    qdrant_api_key=config("qdrantapikey")
    qdrant_url=config("qdrantendpoint")
    client=QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
        )
    vector_store = QdrantVectorStore(client=client, collection_name= collection_name, 
                                 embedding=OpenAIEmbeddings(api_key=config("openaiapikey")))
    retriever = vector_store.as_retriever()
    model = ChatOpenAI(

    model_name="gpt-3.5-turbo",
    openai_api_key=config("openaiapikey"),
    temperature=0)
    
    prompt_template = """
    Answer the question based on the context, in a concise manner, in markdown and using bullet points, giving importance to timelines and locations where applicable.
    Context: {context}
    Question: {question}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = (
        {
            "context":retriever.with_config(top_k=3),
            "question":RunnablePassthrough()
        }
        | RunnableParallel({
            "response": prompt | model,
            "context": itemgetter("context")
        })
    )
    return chain

def storecommonquestionandanswers(id,questionanswersarray):
    commonquestions_collection = db.commonquestions
    commonquestions_data = {
    "id":id,
    "answers":questionanswersarray
    }
    commonquestions_collection.delete_many(filter={"id": id})
    commonquestions = commonquestions_collection.insert_one(commonquestions_data)