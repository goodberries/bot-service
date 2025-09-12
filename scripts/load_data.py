import os
from dotenv import load_dotenv
from langchain_pinecone import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain_aws import BedrockEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import boto3

load_dotenv()

# --- Configuration ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "rag-index"
DATA_FILE = "data/knowledge_base.md"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")


def main():
    """
    Loads documents, splits them, creates embeddings, and stores them in Pinecone.
    """
    if not PINECONE_API_KEY:
        print("PINECONE_API_KEY environment variable is required.")
        return

    print("Loading documents...")
    loader = TextLoader(DATA_FILE)
    documents = loader.load()

    print("Splitting documents...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    print("Initializing Pinecone...")
    pinecone_client = PineconeClient(api_key=PINECONE_API_KEY)

    print("Creating embeddings and indexing in Pinecone...")
    bedrock_client = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    embeddings = BedrockEmbeddings(client=bedrock_client)

    try:
        existing_indexes = [idx["name"] for idx in pinecone_client.list_indexes()]

        if INDEX_NAME not in existing_indexes:
            print(f"Creating new Pinecone index: {INDEX_NAME}")
            embedding_dim = len(embeddings.embed_query(docs[0].page_content))
            pinecone_client.create_index(
                name=INDEX_NAME,
                dimension=embedding_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=AWS_REGION),
            )

            # Store docs in Pinecone
            Pinecone.from_documents(docs, embeddings, index_name=INDEX_NAME)
            print(
                f"Successfully loaded {len(docs)} document chunks into index '{INDEX_NAME}'."
            )
        else:
            print(f"Index '{INDEX_NAME}' already exists. Adding/updating documents...")
            Pinecone.from_documents(docs, embeddings, index_name=INDEX_NAME)
            print(
                f"Successfully upserted {len(docs)} document chunks into index '{INDEX_NAME}'."
            )

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
