# RAG Application for Any File

Welcome to the first version of the RAG (Retrieval-Augmented Generation) application! You can try the application [here](https://ragforanyfile.streamlit.app/). With this application, you can upload any PDF document, with a limit of 200 MB per file. You can upload as many PDFs as you'd like.

Once uploaded, you can engage in a chat-like interface to ask questions about the content of the documents. Whether you're querying about company policies and rules or seeking rules for a board game, you're in control.

## Installation

To install the app scripts, follow these steps:

1. Clone the repository to your local machine using the command:
```git clone https://github.com/pr0fi7/RAG_for_any_file.git```
2. Ensure that you have the required dependencies installed by executing:
```pip install -r requirements.txt```
3. Create a `.env` file with your OpenAI API key in the same directory:
```OPENAI_API_KEY = 'sk-hghghghhhfhfhffhhf'```


## What is RAG?

RAG is a conversational model where you chat with a Language Model (LLM) that has been trained on a specific corpus of data. This allows for contextually relevant responses based on the provided data.

## How it Works

Here's a breakdown of the steps involved, as seen in `app.py`:

1. **Getting PDF Files**: Upload PDF files which are combined into one big text chunk.

2. **Text Preprocessing**: The text chunk is split into smaller chunks for processing.

3. **Creating Vector Store**: A vector store is created to store embedded data and perform vector searches. The application uses Faiss for this, becaus it  runs locally.
![Image Description](assets/rag_indexing-8160f90a90a33253d0154659cf7d453f.png)

4. **Conversational Retrieval Chain**: The application utilizes a Conversational Retrieval Chain mechanism to provide answers based on the provided data. If you want to learn more about it here is the [article](https://medium.com/@jerome.o.diaz/langchain-conversational-retrieval-chain-how-does-it-work-bb2d71cbb665) to explore.

![Image Description](assets/rag_retrieval_generation-1046a4668d6bb08786ef73c56d4f228a.png)

6. **Streamlit Integration**: Everything is combined in Streamlit, providing a chat interface for user interaction. Custom HTML and CSS templates for chat styling can be found in `htmlTemplates.py`.

## Future Changes

For future versions, the following improvements are planned:

- Ability to upload any type of document, not just PDFs.
- Prompt engineering to enhance performance.
- Consideration of alternative embedding tools, such as Instructor, which has better embeddins than OpenAi according to benchmarks.

![1_Mq5tsRyDIaQAZlKGMHqw5g](https://github.com/pr0fi7/RAG_for_any_file/assets/53155116/c4803b7f-ec5e-42e1-9fbd-9c1a6f55c34c)




