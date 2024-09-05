import streamlit as st
import requests

def query_rag(prompt):
    url = 'http://localhost:8000/chat'
    params = {'question': prompt}

    response = requests.get(url, params=params)
    data = response.json()
    return data['reply']


st.title("Testing Interface")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = query_rag(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append(
        {"role": "assistant", "content": response})
