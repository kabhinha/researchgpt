import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import pyttsx3
import tiktoken
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)


def load_document(file):
    _, ext = os.path.splitext(file)
    if ext==".pdf":
        from langchain.document_loaders import PyPDFLoader
        print(f"Loading {file}")
        loader = PyPDFLoader(file)
    elif ext==".docx":
        from langchain.document_loaders import Docx2txtLoader
        print(f"Loading {file}")
        loader = Docx2txtLoader(file)
    elif ext==".txt":
        from langchain.document_loaders import TextLoader
        print(f"Loading {file}")
        loader = TextLoader(file, encoding="UTF-8")
    else:
        print("Format not suported!")
        return None
    data = loader.load()
    return data

# Tell tiktoken what model we'd like to use for embeddings
tiktoken.encoding_for_model('text-embedding-ada-002')

# Intialize a tiktoken tokenizer (i.e. a tool that identifies individual tokens (words))
tokenizer = tiktoken.get_encoding('cl100k_base')

# Create our custom tiktoken function
def tiktoken_len(text: str) -> int:
    """
    Split up a body of text using a custom tokenizer.

    :param text: Text we'd like to tokenize.
    """
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-ada-002')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.0004:.6f}')
    return total_tokens, total_tokens / 1000 * 0.0004


def chunk_data(data, chunk_size=512, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tiktoken_len,
        add_start_index=True
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def create_embeddings(chunks):
    embeddings  = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def ask_with_memory(vector_store, question, chat_history=[], k=3):
    from langchain.chains import ConversationalRetrievalChain
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    crc = ConversationalRetrievalChain.from_llm(llm, retriever)
    result = crc({'question': question, 'chat_history': chat_history})
    chat_history.append((question, result['answer']))

    return result, chat_history

def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain.chat_models import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    answer = chain.run(q)
    return answer


if __name__=="__main__":
    st.subheader("AI Amplifier")

    with st.sidebar:
        uploaded_file = st.file_uploader("Data:", type=["pdf", "txt", "docx"])
        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3)

        # add data button widget
        add_data = st.button('Add Data')
        
        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('/', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=int(chunk_size))
                st.write(f'Chunk size: {int(chunk_size)}, Chunks: {len(chunks)}')


                # creating the embeddings and returning the Chroma vector store
                vector_store = create_embeddings(chunks)
                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store
                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')
                st.success('File uploaded, chunked and embedded successfully.')
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, k)

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)
