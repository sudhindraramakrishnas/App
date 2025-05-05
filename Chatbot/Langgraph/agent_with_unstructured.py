from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper, OpenWeatherMapAPIWrapper
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import os
import logging
import json
from ocr_tool import setup_ocr_tool
from unstructured_pdf_ingestion import setup_unstructured_pdf_ingestion

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def setup_agent_with_unstructured():
    """
    Initialize an agent with search, weather, OCR, and unstructured PDF ingestion capabilities.
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
    
    # Get the OCR and unstructured PDF tools
    ocr_tool = setup_ocr_tool()
    pdf_ingestion_tool = setup_unstructured_pdf_ingestion()

    # Initialize the LLM with the correct model name
    chat_llm = ChatOpenAI(model='gpt-4')

    # Create the agent with all tools
    agent = initialize_agent(
        tools=[search_tool, weather_tool, ocr_tool, pdf_ingestion_tool],
        llm=chat_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent

def process_query(query, file_path=None, file_type=None):
    """
    Process a user query, optionally with a file for OCR or PDF ingestion.
    
    Args:
        query (str): The user's question
        file_path (str, optional): Path to a file (image or PDF)
        file_type (str, optional): Type of file ('image' or 'pdf')
        
    Returns:
        str: The agent's response
    """
    agent = setup_agent_with_unstructured()
    
    try:
        logger.info(f"Processing query: {query}")
        
        # If a file is provided, include appropriate information in the prompt
        if file_path and file_type:
            if file_type.lower() == 'image':
                prompt = f"""Please answer this question: {query}
                
                I've also provided an image at {file_path}. If the question is about the image, 
                use the OCR tool to extract text from it and incorporate that information in your answer.
                """
            elif file_type.lower() == 'pdf':
                prompt = f"""Please answer this question: {query}
                
                I've also provided a PDF file at {file_path}. If the question is about the PDF, 
                use the pdf_ingestion tool to extract structured content from it and incorporate that information in your answer.
                
                The pdf_ingestion tool will extract:
                - Text content
                - Titles and headings
                - Tables
                - Images
                - Page breaks
                
                Use this structured information to provide a comprehensive answer.
                """
            else:
                prompt = f"""Please answer this question: {query}
                
                I've also provided a file at {file_path} of type {file_type}. 
                If the question is about the file, use the appropriate tool to extract information from it.
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

def analyze_pdf_content(pdf_path, output_dir=None):
    """
    Analyze a PDF document and provide a summary of its content.
    
    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save extracted content
        
    Returns:
        dict: Analysis results
    """
    # Initialize the PDF ingestion tool
    pdf_tool = setup_unstructured_pdf_ingestion()
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Ingest the PDF
    result = pdf_tool.run(pdf_path, extract_images=True, output_dir=output_dir)
    
    # Generate a summary of the content
    summary = {
        "file_path": pdf_path,
        "element_counts": {
            "text": len(result["text"]),
            "titles": len(result["titles"]),
            "tables": len(result["tables"]),
            "images": len(result["images"]),
            "pages": len(set([pb["page_number"] for pb in result["page_breaks"]])) if result["page_breaks"] else 0
        },
        "title_list": [title["text"] for title in result["titles"]],
        "sample_text": [text["text"][:200] + "..." for text in result["text"][:5]] if result["text"] else [],
        "image_paths": [img["path"] for img in result["images"] if "path" in img]
    }
    
    # Save the summary to a JSON file if output directory is specified
    if output_dir:
        summary_path = os.path.join(output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")
    
    return summary

# Example usage
if __name__ == "__main__":
    # Example 1: Text-only query
    result1 = process_query("What's the weather in New York?")
    print("Weather query result:")
    print(result1)
    
    # Example 2: Query with image
    # Replace with your image path
    image_path = "path/to/your/image.jpg"
    result2 = process_query("What text is in this image?", image_path, "image")
    print("\nOCR query result:")
    print(result2)
    
    # Example 3: Query with PDF
    # Replace with your PDF path
    pdf_path = "path/to/your/document.pdf"
    result3 = process_query("What is this PDF about?", pdf_path, "pdf")
    print("\nPDF query result:")
    print(result3)
    
    # Example 4: Analyze PDF content
    output_dir = "pdf_analysis"
    summary = analyze_pdf_content(pdf_path, output_dir)
    print("\nPDF Analysis Summary:")
    print(f"Total pages: {summary['element_counts']['pages']}")
    print(f"Total text elements: {summary['element_counts']['text']}")
    print(f"Total titles: {summary['element_counts']['titles']}")
    print(f"Total tables: {summary['element_counts']['tables']}")
    print(f"Total images: {summary['element_counts']['images']}")
    
    if summary["title_list"]:
        print("\nDocument titles:")
        for title in summary["title_list"]:
            print(f"- {title}") 