# from dotenv import load_dotenv
# from langchain_community.utilities import OpenWeatherMapAPIWrapper
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
# from langchain_core.output_parsers import StrOutputParser
# load_dotenv()
# #from langchain.chain import LLMChain
# from langchain.tools import Tool
# from langchain_core.prompts import ChatPromptTemplate
# import os

# os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
# os.environ['OPENWEATHERMAP_API_KEY'] = os.getenv('OPENWEATHERMAP_API_KEY')
# weather = OpenWeatherMapAPIWrapper()
# weather_tool = Tool(
#     name="weather",
#     description="Useful for getting weather information for a specific location",
#     func=weather.run
# )
# search = DuckDuckGoSearchAPIWrapper()

# from langchain_openai import ChatOpenAI
# chat_llm = ChatOpenAI(model='gpt-4o')

# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system","You are an expert in weather. Answer all questions to the best of your ability."),
#         ("human","{weather_input}"),
#     ]
# )

# while True:
#     user_input = input("Please enter weather or news else press q to quit: ")
#     if user_input.lower() =='q':
#         break
#     elif user_input.lower() == 'weather':
#         weather_input = input("Please enter your weather input: ")
#         llm_weather = chat_llm.bind_tools([weather_tool])
#         chain_weather = prompt | llm_weather | StrOutputParser()
#         print(chain_weather.invoke({"weather_input":weather_input}))

from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, OpenWeatherMapAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
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