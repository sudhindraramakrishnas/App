from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, OpenWeatherMapAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import os
import logging
from ocr_tool import setup_ocr_tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_agent_with_ocr():
    """
    Initialize an agent with search, weather, and OCR capabilities.
    """
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
        result = weather.run(query)
        logger.info(f"Weather result: {result}")
        return result
    
    # Create tools
    search_tool = Tool(
        name="search",
        description="Useful for searching the internet for recent news and information. Input should be a search query string.",
        func=search_with_logging
    )
    
    weather_tool = Tool(
        name="weather",
        description="Useful for getting current weather information for a specific location. Input should be a location name.",
        func=weather_with_logging
    )
    
    # Get the OCR tool
    ocr_tool = setup_ocr_tool()

    # Initialize the LLM with the correct model name
    chat_llm = ChatOpenAI(model='gpt-4')

    # Create the agent with all tools
    agent = initialize_agent(
        tools=[search_tool, weather_tool, ocr_tool],
        llm=chat_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent

def process_query(query, image_path=None):
    """
    Process a user query, optionally with an image for OCR.
    
    Args:
        query (str): The user's question
        image_path (str, optional): Path to an image file for OCR
        
    Returns:
        str: The agent's response
    """
    agent = setup_agent_with_ocr()
    
    try:
        logger.info(f"Processing query: {query}")
        
        # If an image is provided, include OCR information in the prompt
        if image_path:
            prompt = f"""Please answer this question: {query}
            
            I've also provided an image at {image_path}. If the question is about the image, 
            use the OCR tool to extract text from it and incorporate that information in your answer.
            """
        else:
            prompt = f"""Please answer this question: {query}
            
            If it's a weather question, use the weather tool.
            If it's a general question, use the search tool.
            """
        
        result = agent.run(prompt)
        logger.info("Query processed successfully")
        return result
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return f"An error occurred: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Example 1: Text-only query
    result1 = process_query("What's the weather in New York?")
    print("Weather query result:")
    print(result1)
    
    # Example 2: Query with image
    # Replace with your image path
    image_path = "path/to/your/image.jpg"
    result2 = process_query("What text is in this image?", image_path)
    print("\nOCR query result:")
    print(result2) 