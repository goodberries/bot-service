from fastapi import FastAPI, HTTPException, Request
from langchain_pinecone import Pinecone
from langchain_aws import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
import os
from dotenv import load_dotenv
import boto3
import sqlalchemy
from sqlalchemy import create_engine, Table, Column, String, MetaData, text, insert
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

interactions = Table('interactions', metadata,
    Column('interaction_id', String, primary_key=True),
    Column('user_query', String),
    Column('bot_response', String),
    Column('timestamp', String),
    Column('feedback', String, nullable=True)
)

@app.on_event("startup")
def startup_event():
    """Create the table on startup if it doesn't exist."""
    with engine.connect() as connection:
        if not sqlalchemy.inspect(engine).has_table("interactions"):
            metadata.create_all(engine)
            print("Created 'interactions' table.")
        else:
            print("'interactions' table already exists.")


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
        
        # Log interaction to the database
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        stmt = insert(interactions).values(
            interaction_id=interaction_id,
            user_query=query,
            bot_response=response_text,
            timestamp=timestamp
        )
        
        with engine.connect() as connection:
            connection.execute(stmt)
            connection.commit()

        return {"response": response_text, "interaction_id": interaction_id}

    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
