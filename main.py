from fastapi import FastAPI, HTTPException
from langchain_pinecone import Pinecone
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Bedrock
import os
from dotenv import load_dotenv
import boto3

load_dotenv()

app = FastAPI()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-index"
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"

# --- Bedrock and Embeddings Clients ---
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("AWS_REGION", "us-east-1")
)
embeddings = BedrockEmbeddings(client=bedrock_client)

# --- Vector Store ---
# Initialize Pinecone and connect to the existing index
vector_store = Pinecone.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embeddings
)

# --- LLM and QA Chain ---
llm = Bedrock(
    client=bedrock_client,
    model_id=BEDROCK_MODEL_ID
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever()
)

@app.post("/process_query")
async def process_query(data: dict):
    """
    Processes a query using the RAG pipeline with Pinecone.
    """
    query = data.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query not provided.")
    
    try:
        response = qa_chain.run(query)
        
        # In Phase 2, we will add logging here
        
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
