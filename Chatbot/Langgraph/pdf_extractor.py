import os
import logging
import fitz  # PyMuPDF
import io
from PIL import Image
from langchain.tools import Tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_pdf_extractor():
    """
    Initialize a tool for extracting text and images from PDF files.
    """
    def extract_from_pdf(pdf_path, extract_images=True, output_dir=None):
        """
        Extract text and images from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            extract_images (bool): Whether to extract images
            output_dir (str, optional): Directory to save extracted images
            
        Returns:
            dict: Dictionary containing extracted text and image paths
        """
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Open the PDF file
            pdf_document = fitz.open(pdf_path)
            
            # Extract text
            text_content = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text_content += page.get_text()
            
            logger.info(f"Extracted {len(text_content)} characters of text")
            
            # Extract images if requested
            image_paths = []
            if extract_images:
                # Create output directory if specified
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                else:
                    # Use the same directory as the PDF
                    output_dir = os.path.dirname(pdf_path)
                
                # Extract images from each page
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    image_list = page.get_images(full=True)
                    
                    for img_index, img in enumerate(image_list):
                        # Get the XREF of the image
                        xref = img[0]
                        
                        # Extract the image bytes
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # Get the image extension
                        image_ext = base_image["ext"]
                        
                        # Create a filename for the image
                        image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                        image_path = os.path.join(output_dir, image_filename)
                        
                        # Save the image
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        image_paths.append(image_path)
                        logger.info(f"Saved image: {image_path}")
            
            # Close the PDF document
            pdf_document.close()
            
            return {
                "text": text_content,
                "images": image_paths,
                "page_count": len(pdf_document)
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            return {
                "error": f"Error processing PDF: {str(e)}",
                "text": "",
                "images": []
            }
    
    # Create a LangChain tool for PDF extraction
    pdf_tool = Tool(
        name="pdf_extractor",
        description="Useful for extracting text and images from PDF files. Input should be a path to a PDF file.",
        func=extract_from_pdf
    )
    
    return pdf_tool

# Example usage
if __name__ == "__main__":
    # Initialize the PDF extractor tool
    pdf_tool = setup_pdf_extractor()
    
    # Example: Extract text and images from a PDF
    pdf_path = "path/to/your/document.pdf"  # Replace with your PDF path
    output_dir = "extracted_images"  # Optional: directory to save images
    
    result = pdf_tool.run(pdf_path, extract_images=True, output_dir=output_dir)
    
    print("Extracted text:")
    print(result["text"][:500] + "..." if len(result["text"]) > 500 else result["text"])
    
    print("\nExtracted images:")
    for img_path in result["images"]:
        print(f"- {img_path}") 