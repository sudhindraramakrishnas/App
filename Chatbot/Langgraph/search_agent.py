from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_search_agent():
    # Set up API keys
    os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

    # Initialize search tool with logging
    search = DuckDuckGoSearchAPIWrapper()
    
    def search_with_logging(query):
        logger.info(f"Search tool invoked with query: {query}")
        result = search.run(query)
        logger.info(f"Search tool returned result of length: {len(result)}")
        return result

    search_tool = Tool(
        name="search",
        description="Useful for searching the internet for recent news and information",
        func=search_with_logging
    )

    # Initialize the LLM
    chat_llm = ChatOpenAI(model='gpt-4')

    # Create the agent with the search tool
    agent = initialize_agent(
        tools=[search_tool],
        llm=chat_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    
    return agent

def search_news(query):
    agent = setup_search_agent()
    try:
        logger.info(f"Starting search for query: {query}")
        result = agent.run(query)
        logger.info("Search completed successfully")
        return result
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    # Example usage
    result = search_news("What is the latest news on ChatGPT")
    print("\nFinal Result:")
    print(result) 