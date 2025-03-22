import os
import json
import requests
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai import APIClient
# Watsonx API Credentials
creds = {
    "api_key": "MY API KEY",
    "url": "https://jp-tok.ml.cloud.ibm.com",
}

# LLM Parameters
parameters = {
    "decoding_method": "greedy",
    "max_new_tokens": 900,
    "min_new_tokens": 1,
    "temperature": 0.5,
    "top_k": 50,
    "top_p": 1,
}

# Initialize Watsonx API Client
api_client = APIClient(creds)

# Create WatsonxLLM instance
watsonx_llm = WatsonxLLM(
    model_id="meta-llama/llama-3-2-11b-vision-instruct",
    project_id="8ea09efa-0308-4442-9ae9-dea6a363127d",
    url = creds["url"],
    apikey=creds["api_key"],
    params=parameters,
)

# Streamlit UI
st.title("📄 What can I assist you today ?")

# File uploader (Users can drag & drop or browse a PDF)
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Check if a PDF is uploaded
if uploaded_file:
    # Save uploaded file
    pdf_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load and index the PDF
    @st.cache_resource
    def load_pdf(file_path):
        loaders = [PyPDFLoader(file_path)]
        index_creator = VectorstoreIndexCreator(
            embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2"),
            text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20),
        )
        return index_creator.from_loaders(loaders)

    # Create an index for the uploaded PDF
    index = load_pdf(pdf_path)
    vectorstore = index.vectorstore

    # Create RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=watsonx_llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )

    st.success(f"✅ PDF '{uploaded_file.name}' successfully uploaded and indexed!")

# Store past messages
if "messages" not in st.session_state:
    st.session_state.messages = []
quick_responses = {
    "hi": "Hello! How can I help you today? 😊",
    "hello": "Hi there! How can I assist you?",
    "hey": "Hey! What do you need help with?",
    "thank you": "You're welcome! Happy to help. 😊",
    "thanks": "No problem! Let me know if you have more questions.",
    "bye": "Goodbye! Have a great day! 👋",
}

# Display previous messages
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

# Chat Input
prompt = st.chat_input("Ask a question...")

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
     # Check for short responses first
    lower_prompt = prompt.lower().strip()
    if lower_prompt in quick_responses:
        response = quick_responses[lower_prompt]
    elif uploaded_file:
        response = chain.run(prompt)  # Use PDF retrieval if file exists
    else:
        response = watsonx_llm.invoke(prompt)  # Use AI model for complex queries
        
    # If a PDF is uploaded, query the document
    if uploaded_file:
        response = chain.run(prompt)
    else:
        response = watsonx_llm.invoke(prompt)

    # Display AI response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
