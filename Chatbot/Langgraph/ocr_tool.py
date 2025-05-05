import easyocr
import logging
from langchain.tools import Tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ocr_tool():
    """
    Initialize the EasyOCR reader and create a tool for OCR functionality.
    """
    # Initialize the OCR reader (first time will download the model)
    # You can specify multiple languages if needed, e.g., ['en', 'fr']
    reader = easyocr.Reader(['en'])
    
    def ocr_with_logging(image_path):
        """
        Perform OCR on an image and return the extracted text.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Extracted text from the image
        """
        try:
            logger.info(f"Processing image: {image_path}")
            # Perform OCR
            results = reader.readtext(image_path)
            
            # Extract text from results
            extracted_text = "\n".join([text for _, text, _ in results])
            
            logger.info(f"Successfully extracted text from image")
            return extracted_text
        except Exception as e:
            logger.error(f"Error during OCR: {str(e)}")
            return f"Error performing OCR: {str(e)}"
    
    # Create a LangChain tool for OCR
    ocr_tool = Tool(
        name="ocr",
        description="Useful for extracting text from images. Input should be a path to an image file.",
        func=ocr_with_logging
    )
    
    return ocr_tool

# Example usage
if __name__ == "__main__":
    # Initialize the OCR tool
    ocr_tool = setup_ocr_tool()
    
    # Example: Extract text from an image
    image_path = "path/to/your/image.jpg"  # Replace with your image path
    result = ocr_tool.run(image_path)
    print("Extracted text:")
    print(result) 