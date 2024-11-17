from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

## Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple QnA Chatbot with Ollama"

## Promot Template
prompt = ChatPromptTemplate.from_messages([
    ("system","You are helpful assistant. Please answer to the queries"),
    ("user","Question:{question}")
])

def generate_response(question,engine):
    llm = OllamaLLM(model=engine)
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({"question":question})
    return answer


## Title
st.title("Chatbot with Open Source Models")

## selecting the OpenAI model
engine = st.sidebar.selectbox("Select a model",["gemma:2b","llama3"])

## main interface for user input
st.write("Ask your question")
user_input = st.text_input("You:")

if user_input:
    response = generate_response(question=user_input,engine=engine)
    st.write(response)
else:
    st.write("Please provide the input")