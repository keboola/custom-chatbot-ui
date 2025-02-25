import os
import logging
from llama_index.core import VectorStoreIndex
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.prompts import Prompt
from llama_index.vector_stores.pinecone import PineconeVectorStore
from langchain.callbacks.streamlit import StreamlitCallbackHandler

import streamlit as st
import openai
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Set page config
st.set_page_config(page_title="Kai - Keboola AI Assistant", page_icon="🤖")

# Configure API keys with Keboola environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
api_key = os.getenv("PINECONE_API_KEY")

# Init Pinecone
pc = Pinecone(api_key=api_key)
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Custom prompt for condensing questions
custom_prompt = Prompt("""\
Given a conversation (between Human and Assistant) and a follow up message from Human, \
rewrite the message to be a standalone question that captures all relevant context \
from the conversation. 

<Chat History> 
{chat_history}

<Follow Up Message>
{question}

<Standalone question>
""")

# Init chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    ai_intro = "Hello, I'm Kai, your Keboola AI Bot. What can I do for you?"
    st.session_state.messages.append({"role": "assistant", "content": ai_intro})

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Init vector store and chat engine
vector_store = PineconeVectorStore(pinecone_index=index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine()
chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    condense_question_prompt=custom_prompt,
    verbose=True
)

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
        
        # Create a container for the callback
        callback_container = st.container()
        st_callback = StreamlitCallbackHandler(callback_container)
        
        # Get response from chat engine
        response = chat_engine.chat(user_input)
        
        # Update placeholder with response
        message_placeholder.markdown(response.response)
        
        # Show sources
        with st.expander("Sources"):
            for i, source_node in enumerate(response.source_nodes):
                st.markdown(f"**Source {i+1}**")
                st.markdown(source_node.node.get_content())
                st.markdown("---")
    
    # Add response to session state
    st.session_state.messages.append({"role": "assistant", "content": response.response})

with st.container():    
    last_output_message = []
    last_user_message = []

    for message in reversed(st.session_state.messages):
        if message["role"] == "assistant":
            last_output_message = message
            break
    for message in reversed(st.session_state.messages):
        if message["role"] =="user":
            last_user_message = message
            break  
