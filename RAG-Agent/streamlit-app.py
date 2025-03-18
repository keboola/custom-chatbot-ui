import os
import logging
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)

# Set page config
st.set_page_config(page_title="RAG Assistant", page_icon="🤖")

# Configure API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Check APIs
if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not set.")
    st.stop()
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set.")
    st.stop()

# Init APIs
client = OpenAI(api_key=OPENAI_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check Index
if INDEX_NAME not in pc.list_indexes().names():
    st.error(f"Error: Index '{INDEX_NAME}' does not exist in your Pinecone account.")
    st.stop()

index = pc.Index(INDEX_NAME)

# Custom prompt for condensing questions
CONDENSE_QUESTION_PROMPT = """
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation. 

<Chat History> 
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
"""

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

def condense_question(chat_history, question):
    """
    Condense a follow-up question based on chat history
    """
    chat_history_text = "\n".join([
        f"{'Human' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
        for msg in chat_history
    ])
    
    prompt = CONDENSE_QUESTION_PROMPT.format(
        chat_history=chat_history_text,
        question=question
    )
    
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=200,
        temperature=0.1
    )
    return response.choices[0].text.strip()

def query_similar_documents(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find documents similar to the query text.
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

def rag_query(query: str, max_tokens: int = 1000) -> Dict:
    """
    Perform a RAG query - retrieve relevant documents and generate a response.
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
    
    return {
        "response": response.choices[0].text.strip(),
        "source_nodes": similar_docs
    }

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    ai_intro = "Hello, I'm your RAG Assistant. What can I help you with?"
    st.session_state.messages.append({"role": "assistant", "content": ai_intro})

# Display title
st.title("RAG Assistant")

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get input
user_input = st.chat_input("Ask me about your data...")

if user_input:
    # Add user message to the chat
    with st.chat_message("user"):
        st.markdown(user_input)
    # Add user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show assistant response with typing indicator
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Process the question with history if needed
        if len(st.session_state.messages) > 2:  # If we have chat history
            standalone_question = condense_question(
                st.session_state.messages[:-1],  # All messages except current
                user_input
            )
            logging.info(f"Original question: {user_input}")
            logging.info(f"Condensed question: {standalone_question}")
        else:
            standalone_question = user_input
        
        # Get response from RAG engine
        response = rag_query(standalone_question)
        
        # Update placeholder with response
        message_placeholder.markdown(response["response"])
        
        # Show sources
        with st.expander("Sources"):
            for i, source_node in enumerate(response["source_nodes"]):
                st.markdown(f"**Source {i+1}**")
                st.markdown(source_node["full_text"])
                st.markdown("---")
                st.markdown("**Metadata:**")
                for key, value in source_node["metadata"].items():
                    st.markdown(f"- **{key}**: {value}")
    
    # Add response to session state
    st.session_state.messages.append({"role": "assistant", "content": response["response"]})

with st.container():    
    last_output_message = []
    last_user_message = []

    for message in reversed(st.session_state.messages):
        if message["role"] == "assistant":
            last_output_message = message
            break
    for message in reversed(st.session_state.messages):
        if message["role"] == "user":
            last_user_message = message
            break 