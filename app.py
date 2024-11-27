import streamlit as st
from test import qa_chain

st.set_page_config(page_title="Hotel Inquiry Chatbot", page_icon="🏨🤖")

st.title("Hotel Inquiry Chatbot 🏨🤖")
st.write("Ask me anything about Lunghton Hotel and Suites, Asaba!")
st.write("Ensure to add a question mark (?) at the end of every question.")

if "history" not in st.session_state:
    st.session_state.history = []

def chat():
    question = st.text_input("You:", key="input")

    if st.button("Send"):
        if question:
            st.session_state.history.append({"role": "user", "content": question})
            chain = qa_chain()
            response = chain(question)
            st.session_state.history.append({"role": "bot", "content": response["result"]})

    for message in st.session_state.history:
        if message["role"] == "user":
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Bot:** {message['content']}")

chat()
