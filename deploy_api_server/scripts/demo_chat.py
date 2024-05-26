import streamlit as st
from openai import OpenAI
import time

st.title("Llama3 Chat Demo")

client = OpenAI(base_url="http://localhost:3000/v1", api_key="None")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

prompt = st.chat_input("Say something")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    for message in st.session_state["messages"]:
        st.chat_message(message["role"]).write(message["content"])
    container = st.empty()
    chat_completion = client.chat.completions.create(
        stream=True,
        messages=st.session_state["messages"],
        model="llama3",  # Must match model name in the Triton server
        max_tokens=512,
    )
    response = ""
    for event in chat_completion:
        content = event.choices[0].delta.content
        if content:
            response += content

        container.chat_message("assistant").write(response)
        time.sleep(0.05)  # Introduce a small delay for rendering

    st.session_state["messages"].append({"role": "assistant", "content": response})
