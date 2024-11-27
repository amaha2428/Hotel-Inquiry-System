# from langchain.llms import GooglePalm
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain

load_dotenv()

# Configure API key
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']
genai.configure(api_key=GOOGLE_API_KEY)

# Define LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.8)

# Use HuggingFaceEmbeddings with model_name specified
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS index file path
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='Lunghton_data.csv', source_column='prompt', encoding='iso-8859-1')
    data = loader.load()

    # Create FAISS vector store
    vector_db = FAISS.from_documents(documents=data, embedding=embedding)
    vector_db.save_local(vectordb_file_path)

def qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, embedding, allow_dangerous_deserialization=True)

    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "Your request is beyond my scope of assessment." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        input_key="query",
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain


if __name__ == '__main__':
    # Uncomment to create vector DB
    # create_vector_db()
    # chain = qa_chain()
    # print('*'*30)
    # response = chain.invoke({'query': 'where is the hotel located?'})
    # print(response['result'])
    # print('*'*30)
    pass
    
