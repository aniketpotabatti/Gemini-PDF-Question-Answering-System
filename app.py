import streamlit as st
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return 

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faisss_index")

def get_conversional_chain():

    prompt_template="""
    Answer the questions as detailed as possible from the provided context, if the answer is not in
    provided context just say, "answer is not available in the given file", Provide accurate answers\n\n
    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:  
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    chain=load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")

    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversional_chain()

    response = chain({"input_documents":docs, "question": user_question} , return_only_outputs=True)
    
    print(response)
    st.write("Reply; ", response["output_text"])
    
def main():
    st.set_page_config("Chat With PDF")
    st.header("Chat with PDF powered by Gemini pro🤖")

    user_question = st.text_input("Ask a Question here")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("1.Upload your files \n\n 2.Click the Submit & Process ", type="pdf", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("processing..."):
                raw_text = ""
                text_chunks = []
                for pdf in pdf_docs:
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        raw_text += page.extract_text()
                

if  __name__ == '__main__':
    main()
