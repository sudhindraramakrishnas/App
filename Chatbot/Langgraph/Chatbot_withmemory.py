from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain_core.messages import HumanMessage, AIMessage
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

# 1. Basic Buffer Memory (stores all conversations)
buffer_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 2. Window Memory (stores last k conversations)
window_memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    k=3,  # number of conversations to remember
    return_messages=True
)

# 3. Summary Memory (maintains a summary of the conversation)
summary_memory = ConversationSummaryMemory(
    memory_key="chat_history",
    llm=llm,
    return_messages=True
)

# Choose which memory type to use
memory = buffer_memory  # or window_memory or summary_memory

# Create a prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant with memory of the conversation."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

# Create the conversation chain
chain = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=True
)

# Function to chat with memory
def chat_with_memory():
    print("Chat started! (Type 'quit' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        # Get the response
        response = chain.invoke({"input": user_input})
        print("\nAI:", response['response'])
        
        # Optionally print the current memory
        print("\nCurrent Memory:")
        print(memory.load_memory_variables({})['chat_history'])

if __name__ == "__main__":
    chat_with_memory()