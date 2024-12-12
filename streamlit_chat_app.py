import streamlit as st
import torch
import time

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate


#### Model initialization
device = 0 if torch.cuda.is_available() else -1

hf = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation",
    device=device, pipeline_kwargs={"max_new_tokens": 200, "pad_token_id": 50256},
)

template = """Question: {question}
Answer:"""
prompt = PromptTemplate.from_template(template)

chain = prompt | hf
####

# Streamed response emulator
def response_generator(user_input):
    response = chain.invoke({"question": user_input}).split("Answer:")[1] # work with template

    for word in response.split():
        yield word + " "
        time.sleep(0.05)


st.title("temp chat")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))

    st.session_state.messages.append({"role": "assistant", "content": response})