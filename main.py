from fastapi import FastAPI, HTTPException, Request
from langchain_pinecone import Pinecone
from langchain_aws import BedrockEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import BedrockChat
import os
from dotenv import load_dotenv
import boto3
import httpx

load_dotenv()

app = FastAPI()

# --- Configuration --- //test
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-index"
BEDROCK_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
INTERACTIONS_SERVICE_URL = os.getenv("INTERACTIONS_SERVICE_URL", "http://interactions-service:8003")


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
    Processes a query, gets a response from the RAG chain,
    and logs the interaction via the interactions-service.
    """
    query = request.query_params.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter not provided.")
    
    try:
        # 1. Get response from RAG chain
        response_text = qa_chain.run(query)
        
        # 2. Log interaction via the interactions-service
        interaction_data = {"user_query": query, "bot_response": response_text}
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{INTERACTIONS_SERVICE_URL}/interactions", json=interaction_data)
            response.raise_for_status() # Raise an exception for bad status codes
            interaction = response.json()
            interaction_id = interaction.get("interaction_id")

        # 3. Return the response and the new interaction_id
        return {"response": response_text, "interaction_id": interaction_id}

    except httpx.HTTPStatusError as e:
        # Log the error and forward the error from the downstream service
        print(f"Error from interactions-service: {e.response.text}")
        raise HTTPException(status_code=e.response.status_code, detail=f"Error from interactions-service: {e.response.text}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred in the bot service.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
