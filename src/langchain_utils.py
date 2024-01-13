import os

import gradio as gr
from langchain.chains import ConversationChain, RetrievalQA, SequentialChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.schema import AIMessage, HumanMessage
from langchain.text_splitter import (CharacterTextSplitter,
                                     RecursiveCharacterTextSplitter)
from langchain.vectorstores import Chroma

from langchain_utils import LLMChain, PromptTemplate


def build_db(embeddings):
    # Build vector db
    loader = DirectoryLoader(".", glob="cv_long.pdf", loader_cls=PyPDFLoader)
    # Load documents from the specified directory using the loader
    documents = loader.load()

    # Create a RecursiveCharacterTextSplitter object to split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # the text will be divided into chunks, and each chunk will contain up to 1000 characters.
        chunk_overlap=125,  # the last 200 characters of one chunk will overlap with the first 200 characters of the next chunk
    )

    # Split documents into text chunks using the text splitter
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()

    # Embed and store the texts
    # Supplying a persist_directory will store the embeddings on disk

    # Define the directory where the embeddings will be stored on disk
    persist_directory = "db"

    # Create a Chroma instance and generate embeddings from the supplied texts
    # Store the embeddings in the specified 'persist_directory' (on disk)
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=persist_directory
    )

    # Persist the database (vectordb) to disk
    vectordb.persist()

    # Set the vectordb variable to None to release the memory
    vectordb = None


def construct_chain(vectordb):
    # construct chain
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    llm = HuggingFaceHub(
        repo_id=model_id, model_kwargs={"temperature": 0.1, "max_new_tokens": 64}
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3}, search_type="mmr")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        output_key="cv_context",
        return_source_documents=False,
    )

    qa_chain.combine_documents_chain.llm_chain.prompt.template = """
    <s>[INST] Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, 
    don't try to make up an answer.

    {context}

    Question: {question} [/INST] 
    Helpful Answer:</s>
    """

    conversation_prompt = """<s>[INST] You are an AI having a conversation with Human.
    You can see the historical conversation under current conversation.
    Provide an answer to the last Human input based on information from context and conversation history.

    If the AI does not know the answer to a question, it truthfully says it does not know.

    Current conversation:
    {history}
    Human: {input}
    Context: {cv_context} [/INST]
    Helpful answer:</s>
    [INST] only show AI's response as an answer [/INST]
    """

    chat_memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=False,
        input_key="input",
    )

    conversation_chain = ConversationChain(llm=llm, memory=chat_memory, verbose=True)

    conversation_chain.prompt = PromptTemplate(
        input_variables=["history", "input", "cv_context"], template=conversation_prompt
    )

    # This is the overall chain where we run these two chains in sequence.
    overall_chain = SequentialChain(
        chains=[qa_chain, conversation_chain],
        input_variables=["input", "query"],
        verbose=True,
    )

    return overall_chain, chat_memory
