import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.chains import RetrievalQA
import gradio as gr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Suppress warnings
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

## LLM - Using OpenAI
def get_llm():
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=256
    )
    return llm

## Document loader
def document_loader(file):
    loader = PyPDFLoader(file)
    loaded_document = loader.load()
    return loaded_document

## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

## Embedding model - Using OpenAI
def watsonx_embedding():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    return embedding_model

## Vector db
def vector_database(chunks):
    embedding_model = watsonx_embedding()
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_model)
    return vectordb

## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever_obj, 
        return_source_documents=False
    )
    response = qa.invoke(query)
    return response['result']

# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Input Query", lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label="Output", lines=15, max_lines=30),  # Starts at 15 lines, expands to 30
    title="QA bot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

# Launch the app
rag_application.launch(server_name="127.0.0.1", server_port=7860)