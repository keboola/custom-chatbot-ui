import os
import logging
import csv
import json
import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from dotenv import load_dotenv
from typing import List, Dict, Any
from keboola.component import CommonInterface, dao


load_dotenv()

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")

ci = CommonInterface()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not set.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)

vector_store = st.sidebar.radio(
    "Select Vector Store",
    ["Pinecone", "Keboola"],
    index=0  # Default to Pinecone
)

if vector_store == "Pinecone":
    if not PINECONE_API_KEY:
        st.error("PINECONE_API_KEY not set.")
        st.stop()
    if not INDEX_NAME:
        st.error("PINECONE_INDEX_NAME not set.")
        st.stop()
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if INDEX_NAME not in pc.list_indexes().names():
        st.error(f"Error: Index '{INDEX_NAME}' does not exist in your Pinecone account.")
        st.stop()
    
    index = pc.Index(INDEX_NAME)
    st.sidebar.success("Connected to Pinecone")

elif vector_store == "Keboola":
    try:
        tables = ci.get_input_tables_definitions()
        if not tables:
            st.error("No input tables configured in Keboola.")
            st.stop()
        st.sidebar.success("Connected to Keboola")
    except Exception as e:
        st.error(f"Error connecting to Keboola: {e}")
        st.stop()

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
    Find documents similar to the query text using the selected vector store.
    """
    query_embedding = create_embedding(query_text)
    
    if vector_store == "Pinecone":
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
    
    elif vector_store == "Keboola":
        table = ci.get_input_files_definitions()[0]
        
        with open(table.full_path, 'r') as f:
            # CSV format with columns: text, metadata, embedding
            reader = csv.DictReader(f)
            rows = list(reader)
        
        from scipy.spatial.distance import cosine
        similar_docs = []
        
        for row in rows:
            # Convert string embedding to list of floats
            row_embedding = [float(x) for x in row['embedding'].strip('[]').split(',')]
            similarity = 1 - cosine(query_embedding, row_embedding)
            
            similar_docs.append({
                "id": str(hash(row['text'])),  # Generate an ID from the text content
                "score": similarity,
                "text_preview": row['text'][:100] + "..." if len(row['text']) > 100 else row['text'],
                "full_text": row['text'],
                "metadata": json.loads(row['metadata']) if row.get('metadata') else {}
            })
        
        similar_docs.sort(key=lambda x: x['score'], reverse=True)
        return similar_docs[:top_k]

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