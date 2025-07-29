from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_pinecone import PineconeVectorStore
from langchain.chains.retrieval import create_retrieval_chain
from dotenv import load_dotenv
import streamlit as st
from pinecone import Pinecone
import time
import os
load_dotenv()
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
prompt_templete = """
<prompt>
  <role>Metaviz Information Assistant</role>
  <description>
    RAG-based chatbot that provides details about Metaviz by retrieving data and generating natural, informative responses.
  </description>
  <goals>
    <primary>Retrieve raw information and manipulate it into useful answers.</primary>
    <secondary>Maintain a casual, human-like tone and provide relevant, accurate information.</secondary>
  </goals>
  <instructions>
    <step>Analyze the user's message to understand intent.</step>
    <step>Retrieve relevant data from the vector database based on the query.</step>
    <step>Manipulate the retrieved data into a natural, easy-to-read format.</step>
    <step>Always provide accurate, fact-based information.</step>
    <step>If no relevant data is found, apologize and let the user know politely.</step>
  </instructions>
  <Style>
    <tone>Conversational</tone>
    <formality>Casual yet professional</formality>
    <output>Clear, informative, and aligned with Metaviz’s branding</output>
  </Style>
  <Context>{context}</Context>
  <UserInput>Here is the user input: {input}</UserInput>
</prompt>
"""

class MetavizChatBot:
    def __init__(self, prompt_template: str):
        self.prompt_template = prompt_template
        self.chain = self.create_chain()

    def load_vector_store(self):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        pc = Pinecone()
        index = pc.Index("metaviz-knowledge-base-semantic")

        vector_store = PineconeVectorStore(index=index, embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        return retriever
    

    def create_prompt_template(self):
        return ChatPromptTemplate.from_template(self.prompt_template)

    def create_llm(self):
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)

    def create_chain(self):
        doc_chain = create_stuff_documents_chain(llm=self.create_llm(), prompt=self.create_prompt_template())
        retriever = self.load_vector_store()
        return create_retrieval_chain(retriever, doc_chain)

    def invoke_chain(self, query: str):
        try:
            response = self.chain.invoke({"input": query})
            return response.get("answer", "No answer returned.")
        except Exception as e:
            return f"Error during query: {e}"




if "bot" not in st.session_state:
    st.session_state.bot = MetavizChatBot(prompt_template=prompt_templete)
st.set_page_config(page_title="Chat App", page_icon="💬")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Title ---
st.title("💬 Metaviz Chat App")
st.header("Ask anything about Metaviz")

# --- Show Existing Messages ---
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)

# --- User Input ---
user_input = st.chat_input("Type your message...")

# --- When User Sends a Message ---
if user_input:
    resp = st.session_state.bot.invoke_chain(user_input)
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Simulate a response (you can replace this logic later with LLM)
    assistant_response = f"{resp}"

    # Save assistant's message
    st.session_state.chat_history.append(("assistant", assistant_response))
    with st.chat_message("assistant"):
        st.markdown(assistant_response)
