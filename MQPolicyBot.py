import json
import os

# Disable parallelism for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import requests
from llama_index.core import VectorStoreIndex, ServiceContext, Document
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.core.service_context_elements.llm_predictor import LLMPredictor
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from docx import Document

from PyPDF2 import PdfReader
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
#embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")

#Trying to run llm hosted on HuggingFace:

#Possible questions to ask:
# What is contract cheating?
# What are the password policies at the university?


llm_predictor = LLMPredictor(
        llm=Ollama(
            temperature= 0.4,
            model="mistral",
            context_window= 4096,
        ),
        
#          llm=mistral_llm
 )

st.set_page_config(page_title="MQ Policy Assistant", page_icon="🔍", layout="centered", initial_sidebar_state="auto", menu_items=None)


# Title and logo
col1, col2 = st.columns([3, 1])
with col1:
    st.title('MQ Policy Assistant 🔍')
with col2:
    st.image("./image/MQLogo.png", use_column_width=True)

       
# Initialize chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about the university's academic and policy documents!"}
    ]

# Load preloaded documents
def load_data():
    with st.spinner(text="Loading and indexing university policy documents – hang tight!"):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()

        service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            prompt_helper="""
                **You are a large language model trained to assist with queries related to the university's academic and policy documents.**

                **In addition to your general training, you have been provided with a set of specific documents relevant to the university.**
                **These documents include:**

                * {% for doc in docs %}
                    * {{doc.metadata.get('file_name') }}
                {% endfor %}

                Your primary function is to answer user queries related to academic policies, administrative procedures, and other university-related information, considering both your general training and the provided documents.

                **Assume all user questions are relevant to the university context.** If a question seems unrelated, analyze it to identify any potential connections and provide the most relevant information you can, referencing the provided documents where applicable.

                **When responding, prioritize factual accuracy and avoid making claims that haven't been verified through reliable sources.** Use your technical knowledge to explain complex concepts in a clear and concise way.

                **Maintain a respectful and objective tone throughout your interactions.** Avoid expressing personal opinions or beliefs, and focus solely on providing informative and helpful responses.

                **Do not reveal any information about external sources or documents used during your processing.** Keep the conversation focused on directly answering the user's query.

                **Always strive to improve your understanding of university policies and academic procedures, incorporating new knowledge as it becomes available.** This will enable you to provide increasingly accurate and comprehensive responses.

                **When unsure about an answer, acknowledge your limitations and suggest alternative information sources or clarify the user's question.**

                **You are still under development, and your responses may not always be perfect.** Please provide feedback to your developers so they can continue to improve your capabilities.
            """,
            embed_model=embed_model
        )

        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

# Load data without file uploading
index = load_data()

if "chat_engine" not in st.session_state.keys() or st.session_state.get("previous_index") != index:
    st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
    st.session_state["previous_index"] = index  # Store the current index for comparison

if prompt := st.chat_input("Your question", key="user_input"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
