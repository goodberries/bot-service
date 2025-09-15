from fastapi import FastAPI, HTTPException, Request
from langchain_pinecone import Pinecone
from langchain_aws import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
import os
from dotenv import load_dotenv
import boto3
import sqlalchemy
from sqlalchemy import create_engine, Table, Column, String, MetaData, insert, Integer, DateTime
from sqlalchemy.dialects.postgresql import UUID as pgUUID
import uuid
from datetime import datetime

load_dotenv()

app = FastAPI()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-index"
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
DATABASE_URL = os.getenv("DATABASE_URL")



# --- Database Setup ---
engine = create_engine(DATABASE_URL)
metadata = MetaData()

# This is now the single source of truth for the table schema.
interactions = Table('interactions', metadata,
    Column('interaction_id', pgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
    Column('user_query', String),
    Column('bot_response', String),
    Column('timestamp', DateTime, default=datetime.utcnow),
    Column('feedback', Integer, nullable=True) # Stores 1 for like, -1 for dislike
)

@app.on_event("startup")
def startup_event():
    """Create the table on startup if it doesn't exist."""
    try:
        with engine.connect() as connection:
            if not sqlalchemy.inspect(engine).has_table("interactions"):
                metadata.create_all(engine)
                print("Created 'interactions' table.")
                print(DATABASE_URL)
            else:
                print("'interactions' table already exists.")
                print(DATABASE_URL)
    except Exception as e:
        print(f"Database connection failed during startup: {e}")

# --- Bedrock and Embeddings Clients ---
bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
embeddings = BedrockEmbeddings(client=bedrock_client)

# --- Vector Store ---
vector_store = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# --- LLM and QA Chain ---
llm = BedrockChat(
    client=bedrock_client,
    model_id=BEDROCK_MODEL_ID
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

@app.post("/chat")
async def chat(request: Request):
    """
    Processes a query, logs it, and returns a response with an interaction_id.
    """
    query = request.query_params.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter not provided.")
    
    try:
        # Get response from RAG chain
        response_text = qa_chain.run(query)
        
        # Log interaction to the database, returning the generated ID
        stmt = insert(interactions).values(
            user_query=query,
            bot_response=response_text,
        ).returning(interactions.c.interaction_id)
        
        with engine.connect() as connection:
            result = connection.execute(stmt)
            connection.commit()
            interaction_id = result.fetchone()[0]

        return {"response": response_text, "interaction_id": str(interaction_id)}

    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
