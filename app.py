import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS #local one so everything will be deleted after the session
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template
import docx
from pptx import Presentation
import glob
from link_parser import LinkParser

link_parser = LinkParser()

def getText(filename):
    doc = docx.Document(filename[0])
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)

def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_pptx_text(pptx_docs):
    prs = Presentation(pptx_docs[0])
    fullText = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text"):
                fullText.append(shape.text)
    return '\n'.join(fullText)

def get_text_from_url(url):
    try:
        extracted_text = link_parser.extract_text_from_website(url)
        return extracted_text
    except Exception as e:
        st.error(f"An error occurred: {str(e)}. Please enter a valid URL.")
        return None

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1200, chunk_overlap=200)
    text_chunks = splitter.split_text(text)
    return text_chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vector_store

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    api_key = st.secrets["OPENAI_API_KEY"]
    st.set_page_config(page_title="RAG", page_icon=":shark:")
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state or st.session_state.conversation is None:
        st.session_state.conversation = None
    st.header("RAG Chatbot with any document or URL :shark:") 
    
    message = st.text_input("Ask a question")

    if message:
        handle_userinput(message)

    with st.sidebar:
        st.subheader("Your documents")
        st.subheader('Current version, supports PDF, DOCX, PPTX, and URL')
        
        pdf_docs = st.file_uploader("Upload a document", accept_multiple_files=True)
        url = st.text_input("Enter a URL")

        st.subheader("Manual")
        st.write("1. Upload a PDF, DOCX, or PPTX file or provide a URL for analysis.")
        st.write('Note: you can upload multiple files at once as well as combine different file types and content from URLs.')
        st.write("2. Click the 'Analyze' button to start the analysis.")
        st.write("3. Ask a question in the chatbox to get started.")
        st.write("4. The chatbot will provide answers based on the content of the uploaded document or URL.")

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                if pdf_docs is not None and len(pdf_docs) > 0:  # Check if pdf_docs is not None and not empty
                    if pdf_docs[0].name[-3:] == "pdf":
                        raw_text = get_pdf_text(pdf_docs)
                    elif pdf_docs[0].name[-4:] == "docx":
                        raw_text = getText(pdf_docs)
                    elif pdf_docs[0].name[-4:] == "pptx":
                        raw_text = get_pptx_text(pdf_docs)
                    st.session_state.conversation = None  # Reset conversation
                elif url:
                    raw_text = get_text_from_url(url)
                    st.session_state.conversation = None  # Reset conversation
                else:
                    st.error("Please upload a PDF, DOCX, or PPTX file or provide a URL for analysis.")

                if raw_text is None:
                    st.stop()

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vector_store(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore) 

                st.write("Analysis complete")
         
if __name__ == '__main__':
    main()