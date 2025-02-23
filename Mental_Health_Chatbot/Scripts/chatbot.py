import streamlit as st
from generate_response import generate_response


# 🔹 Streamlit UI
st.set_page_config(page_title="AI Mental Health Chatbot", layout="wide")

st.title("🧠 AI Mental Health Chatbot")
st.write("I'm here to listen. How are you feeling today?")

# 🔹 Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔹 User Input Box
user_input = st.text_input("You:", "", key="input")

if st.button("Send"):
    if user_input:
        # Append user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Generate chatbot response
        bot_response = generate_response(user_input)

        # Append bot response to chat history
        st.session_state.chat_history.append({"role": "bot", "content": bot_response})

# 🔹 Display Chat History
st.write("### Chat History")
for chat in st.session_state.chat_history:
    role = "🧑‍💻 You" if chat["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {chat['content']}")

# 🔹 Option to Reset Chat
if st.button("Reset Chat"):
    st.session_state.chat_history = []
    st.experimental_rerun()
