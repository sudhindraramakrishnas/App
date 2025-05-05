from dotenv import load_dotenv
load_dotenv()
import os

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

from langchain_openai import ChatOpenAI
chat_llm = ChatOpenAI(model='gpt-4o')

while True:
    user_input = input("Please enter your input else press q to quit: ")
    if user_input.lower() =='q':
        break
    else:
        print(chat_llm.invoke(user_input).content)
