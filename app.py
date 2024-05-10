import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS #local one so everything will be deleted after the session
from langchain_text_splitters import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from htmlTemplates import css, bot_template, user_template

# model = INSTRUCTOR('hkunlp/instructor-xl')
def get_pdf_text(pdf_docs):

    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1200, chunk_overlap=200)
    text_chunks = splitter.split_text(text)
    return text_chunks


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = model.encode([['Represent the sentence', text_chunks]])
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
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    api_key = os.getenv("OPENAI_API_KEY")
    st.set_page_config(page_title="RAG", page_icon=":shark:")
    st.write(css, unsafe_allow_html=True)

    if 'conversation' not in st.session_state or st.session_state.conversation is None:
        st.session_state.conversation = None
    st.header("RAG")
    
    message = st.text_input("Ask a question")

    if message:
        handle_userinput(message)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload a document", accept_multiple_files=True)

        if st.button("Analyze"):
            with st.spinner("Analyzing..."):
                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vector_store(text_chunks)
                    # Load OpenAI embedding model
                # embeddings = OpenAIEmbeddings()

                # # Load OpenAI chat model
                # llm = ChatOpenAI(temperature=0)

                # # Load the local vector database as a retriever
                # vector_store = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
                # retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                st.session_state.conversation = get_conversation_chain(vectorstore) 

                st.write("Analysis complete")
        

    
if __name__ == '__main__':
    main()