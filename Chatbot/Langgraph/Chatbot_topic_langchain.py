# from dotenv import load_dotenv
# from langchain_community.utilities import OpenWeatherMapAPIWrapper
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# from langchain_core.output_parsers import StrOutputParser
# load_dotenv()
# from langchain.tools import Tool
# from langchain_core.prompts import ChatPromptTemplate
# import os

# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# os.environ['OPENWEATHERMAP_API_KEY'] = os.getenv('OPENWEATHERMAP_API_KEY')

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system","You are an expert in Supply chain. Answer all questions to the best of your ability."),
#         ("user","{user_input}"),
#     ]
# )

# from langchain_openai import ChatOpenAI
# chat_llm = ChatOpenAI(model='gpt-4o')

# while True:
#     user_input = input("Please enter topic on Supply chain or else press q to quit: ")
#     if user_input.lower() =='q':
#         break
#     else :
#         #llm_news = chat_llm.bind_tools([search])
#         chain_weather = prompt | chat_llm | StrOutputParser()
#         print(chain_weather.invoke({"user_input":user_input}))


#Use the Tool calling agent to merge the prompt and the tool in langchain.
#You cannot merge prompt and tool in langchain in a normal way. Use the create_tool_calling_agent to do it and 
# then use the AgentExecutor to run it.

from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, OpenWeatherMapAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType, create_tool_calling_agent
from langchain_openai import ChatOpenAI
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_search_agent():
    # Set up API keys
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    # Initialize search and weather tools
    search = DuckDuckGoSearchAPIWrapper()
    weather = OpenWeatherMapAPIWrapper()
    
    def search_with_logging(query):
        logger.info(f"Searching for: {query}")
        result = search.run(query)
        logger.info(f"Search result: {result}")
        return result
    
    def weather_with_logging(query):
        logger.info(f"Getting weather for: {query}")
        print("Running the weather tool")
        result = weather.run(query)
        logger.info(f"Weather result: {result}")
        return result
    
    search_tool = Tool(
        name="search",
        description="Useful for searching the internet for recent news and information",
        func=search_with_logging
    )
    
    weather_tool = Tool(
        name="weather",
        description="Useful for getting current weather information for a specific location",
        func=weather_with_logging
    )

    # Initialize the LLM
    chat_llm = ChatOpenAI(model='gpt-4o')

    # Create the agent with both tools
    agent = initialize_agent(
        tools=[search_tool, weather_tool],
        llm=chat_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

def search_news(query):
    agent = setup_search_agent()
    try:
        logger.info(f"Starting search for query: {query}")
        result = agent.run(
            f"""Please use the appropriate tool to answer this question: {query}
            If it's a weather question, use the weather tool.
            If it's a general question, use the search tool."""
        )
        logger.info("Search completed successfully")
        return result
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    # Example usage
    user_input = input("Enter your query: ")
    result = search_news(user_input)
    print(result)