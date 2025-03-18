import os
from dotenv import load_dotenv
from pinecone import Pinecone
from openai import OpenAI
from typing import List, Dict, Any
import uuid
import textwrap

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Check APIs
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not set.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set.")

# Init APIs
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check Index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Error: Index '{INDEX_NAME}' does not exist in your Pinecone account.")
    exit(1)

index = pc.Index(INDEX_NAME)

# Create embedding to compare
def create_embedding(text: str) -> List[float]:
    """
    Create a vector embedding from text using OpenAI's embedding model.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def query_similar_documents(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find documents similar to the query text.
    
    Args:
        query_text: The text to find similar documents for
        top_k: Number of similar documents to return
        
    Returns:
        List of similar documents with their metadata and scores
    """
    query_embedding = create_embedding(query_text)
    
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    similar_docs = []
    for match in results["matches"]:
        metadata = match.get("metadata", {})        
        text_content = metadata.get("text", "")
        
        similar_docs.append({
            "id": match["id"],
            "score": match["score"],
            "text_preview": text_content,
            "full_text": text_content,
            "metadata": {k: v for k, v in metadata.items() if k != "text"}
        })
    
    return similar_docs

def rag_query(query: str, max_tokens: int = 1000) -> str:
    """
    Perform a RAG query - retrieve relevant documents and generate a response.
    
    Args:
        query: The user's question
        max_tokens: Maximum tokens for the generated response
        
    Returns:
        Generated response based on retrieved documents
    """
    similar_docs = query_similar_documents(query, top_k=3)
    
    context = "\n\n".join([f"Document {i+1}:\n{doc['full_text']}" 
                         for i, doc in enumerate(similar_docs)])
    
    prompt = f"""
    Based on the following retrieved documents, please answer the query.
    Query: {query}
    Retrieved Documents:
    {context}
    Answer:
    """
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.7
    )
    return response.choices[0].text.strip()


def demo():
    print("\n=== RAG Agent Demo ===\n")
    
    index_stats = index.describe_index_stats()
    while True:
        query = input("\nEnter your question ('q' to quit): ")
        
        if query.lower() in ["quit", "q"]:
            break
            
        print("\nSearching for relevant docs")
        similar_docs = query_similar_documents(query)
        
        print("\nRetrieved docs:")
        for i, doc in enumerate(similar_docs):
            print(f"\n{i+1}. Score: {doc['score']:.4f}")
            print(f"   Preview: {doc['text_preview']}")
        
        print("\nGenerating answer...")
        answer = rag_query(query)
        
        print("\n" + textwrap.fill(answer, width=80))



if __name__ == "__main__":
    print("Starting RAG agent with Pinecone...")
    demo()
