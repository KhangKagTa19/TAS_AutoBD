from tavily import TavilyClient
from langchain.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import List
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain import PromptTemplate, FewShotPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import BSHTMLLoader
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain.chains import LLMChain
from typing import List
from langchain_core.runnables import RunnablePassthrough
from langchain import hub
import getpass
from langchain_community.vectorstores import FAISS
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
import tiktoken
import regex as re
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
import json
import wget
import time
import csv
import requests
import math
import zipfile
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
import random
import torch
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from langchain_experimental.text_splitter import SemanticChunker
import asyncio
import aiohttp
import streamlit as st
from PIL import Image
import base64
from io import BytesIO
from email import send_email  
from make_db import make_db  
from get_hypo import get_hypothesis_idea 
from get_proposal import make_proposal 
from get_info import get_company_information  


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

os.environ['OPENAI_API_KEY'] = "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"

llm = ChatOpenAI(model="gpt-4o", temperature=1, api_key=os.environ["OPENAI_API_KEY"])

encoder = OpenAIEmbeddings(model='text-embedding-3-large') 
#     encoder = HuggingFaceEmbeddings()
tokenizer = tiktoken.get_encoding('cl100k_base')

faiss_index_path = "/kaggle/working/faiss_index"

# Check if the FAISS index exists
if os.path.exists(faiss_index_path):
    # Load the FAISS index if it exists
    print("FAISS index found, loading it...")
    docsearch = FAISS.load_local(faiss_index_path, embeddings=encoder, allow_dangerous_deserialization=True)

# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)
tiktoken.encoding_for_model('text-embedding-3-large')

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2800,  # Reduced chunk size for faster processing
    chunk_overlap=20,  # Reduced overlap to minimize redundancy
    length_function=tiktoken_len,
    separators=['\n\n', '\n', ' ', '']
)

semantic_text_splitter = SemanticChunker(
    OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
)

print("START SESSION")

# Load your embedding model from HuggingFace (LangChain's requirement)
embedding_model = llm


async def get_company_information_async(company_name, another_url):
        return get_company_information(company_name, another_url)


async def main():  # Make the main function asynchronous
    # Set page config for a wider layout
    st.set_page_config(layout="wide")

    # Custom CSS for white background, purple text, and purple input fields with white text
    st.markdown(""" 
    <style>
    /* Main app styles */
    .stApp {
        background-color: #FFFFFF;
        color: #6A0DAD;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    .stButton>button {
        background-color: #FFFFFF !important;
        color: #6A0DAD !important;
        border-color: #6A0DAD !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    .stTextInput>div>div>input {
        background-color: #6A0DAD !important;
        color: #FFFFFF !important;
        border-color: #FFFFFF !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    .stTextArea>div>div>textarea {
        background-color: #6A0DAD !important;
        color: #FFFFFF !important;
        border-color: #FFFFFF !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    .stSelectbox>div>div>div {
        background-color: #FFFFFF !important;
        color: #6A0DAD !important;
        border-color: #6A0DAD !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    h1, h2, h3, p, label {
        color: #6A0DAD !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    .logo-container {
        position: fixed;
        bottom: 20px;
        right: 20px;
        z-index: 999;
    }
    /* Adjust other Streamlit elements */
    .stAlert {
        background-color: #FFFFFF !important;
        color: #6A0DAD !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    .stProgress > div > div > div > div {
        background-color: #6A0DAD !important; /* Changed to purple */
        color: #FFFFFF !important; /* Set text color to white */
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    .stProgress > div > div > div > div > span {
        color: #FFFFFF !important; /* Ensure text in progress bar is white */
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    /* Ensure dropdown text is purple */
    .stSelectbox>div>div>div>div>div {
        color: #6A0DAD !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }

    /* Sidebar styles */
    [data-testid="stSidebar"] {
        background-color: #6A0DAD !important; /* Changed to purple */
    }
    [data-testid="stSidebar"] .stMarkdown p {
        color: #FFFFFF !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: #FFFFFF !important;
        color: #6A0DAD !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    [data-testid="stSidebar"] .stTextInput>div>div>input {
        background-color: #FFFFFF !important;
        color: #6A0DAD !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    [data-testid="stSidebar"] .stSelectbox>div>div>div {
        background-color: #FFFFFF !important;
        color: #6A0DAD !important;
        font-family: 'Trebuchet MS', sans-serif; /* Set font to Trebuchet MS */
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AutoBD")

    # Initialize session state variables
    if 'step' not in st.session_state:
        st.session_state.step = 1
    if 'company_name' not in st.session_state:
        st.session_state.company_name = ""
    if 'characteristics' not in st.session_state:
        st.session_state.characteristics = ""
    if 'idea' not in st.session_state:
        st.session_state.idea = ""
    if 'keywords' not in st.session_state:
        st.session_state.keywords = []
    if 'email_list' not in st.session_state:
        st.session_state.email_list = []
    if 'docsearch' not in st.session_state:
        st.session_state.docsearch = None
    if 'email_proposal' not in st.session_state:
        st.session_state.email_proposal = ""
    if 'logo_base64' not in st.session_state:
        # Load and encode the logo image only once
        logo = Image.open("/kaggle/input/d/kagdeptry/logotas/TAS.png")
        buffered = BytesIO()
        logo.save(buffered, format="PNG")
        st.session_state.logo_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Add logo to the bottom right corner using the stored base64 image
    st.markdown(
        f"""
        <div class="logo-container">
            <img src="data:image/png;base64,{st.session_state.logo_base64}" width="100">
        </div>
        """,
        unsafe_allow_html=True
    )

    # Step 1: Enter Company Name
    if st.session_state.step == 1:
        st.subheader("Step 1: Enter Company Name")
        company_name = st.text_input("Enter Target Company Name", st.session_state.company_name)

        # New input for additional URL
        additional_url = st.text_input("Add an additional URL (optional):")

        if company_name != st.session_state.company_name:
            st.session_state.company_name = company_name

        if st.button("Fetch Company Information"):
            with st.spinner("Fetching company information..."):
                info, email_list = await get_company_information_async(st.session_state.company_name, additional_url)  # Await the async function
                st.session_state.characteristics = info.content
                st.session_state.email_list = email_list


                # Automatically move to Step 2 after fetching information
                st.session_state.step = 2
            st.rerun()

    # Step 2: Review and Edit Company Characteristics
    elif st.session_state.step == 2:
        st.subheader("Step 2: Review Company Characteristics")
        characteristics = st.text_area("Characteristics", st.session_state.characteristics, height=200)

        if characteristics != st.session_state.characteristics:
            st.session_state.characteristics = characteristics

        if st.button("Generate Idea"):
            with st.spinner("Generating idea..."):
                hypo, keywords = await get_hypothesis_idea(st.session_state.characteristics)  # Await the async function
                st.session_state.idea = hypo.content
                st.session_state.keywords = keywords
                st.session_state.step = 3
            st.rerun()


    # Step 3: Review and Edit Generated Idea
    elif st.session_state.step == 3:
        st.subheader("Step 3: Review Generated Idea")
        idea = st.text_area("Idea", st.session_state.idea, height=200)

        if idea != st.session_state.idea:
            st.session_state.idea = idea

        if st.button("Generate Proposal"):
            with st.spinner("Generating proposal..."):
                st.session_state.docsearch = await make_db(st.session_state.idea, st.session_state.keywords)  # Await the async function
                st.session_state.email_proposal = make_proposal(st.session_state.idea, st.session_state.docsearch, st.session_state.company_name)
                st.session_state.step = 4
            st.rerun()

    # Step 4: Review and Edit Email Proposal
    elif st.session_state.step == 4:
        st.subheader("Step 4: Review Email Proposal")
        modified_proposal = st.text_area("Proposal", st.session_state.email_proposal, height=300)

        if modified_proposal != st.session_state.email_proposal:
            st.session_state.email_proposal = modified_proposal

        st.subheader("Select Email")

        st.session_state.email_list = add_email_manually(st.session_state.email_list)

        selected_email = st.selectbox("Choose an email", st.session_state.email_list)

        if st.button("Send Email"):
            with st.spinner("Sending email..."):
                success, message = send_email(selected_email, "Business Proposal", remove_first_last_line(remove_first_last_line(st.session_state.email_proposal)))
                if success:
                    st.success(f"Email sent to {selected_email}")
                    st.balloons()
                    # Clear session state to restart
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]  # Clear all session state
                    st.session_state.step = 1  # Reset to step 1
                else:
                    st.error(f"Failed to send email: {message}")
            st.rerun()

    # Display current step (for debugging)
    st.sidebar.write(f"Current Step: {st.session_state.step}")


if __name__ == '__main__':
    asyncio.run(main())