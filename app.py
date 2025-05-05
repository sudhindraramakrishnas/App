import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Page config
st.set_page_config(page_title="Chatbot with Memory", page_icon="🤖")
st.title("🤖 Chatbot with Memory")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for memory type
if "memory_type" not in st.session_state:
    st.session_state.memory_type = "buffer"

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatOpenAI(model="gpt-4o")

llm = get_llm()

# Initialize different memory types
@st.cache_resource
def get_memories():
    return ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

memory = get_memories()

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant with memory of the conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Initialize conversation chain
@st.cache_resource
def get_chain():
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=prompt,
        verbose=True
    )
    return chain

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if user_input := st.chat_input("What's on your mind?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    chain = get_chain()

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke({"input": user_input})
            st.markdown(response['response'])
            
            # Add assistant message to chat history
            st.session_state.messages.append({"role": "assistant", "content": response['response']})