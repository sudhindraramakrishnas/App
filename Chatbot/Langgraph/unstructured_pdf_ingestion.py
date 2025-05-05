import os
import logging
from typing import Dict, List, Any, Optional
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Text, Image, Table, Title, NarrativeText
from langchain.tools import Tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_unstructured_pdf_ingestion():
    """
    Initialize a tool for ingesting multimodal PDF documents using unstructured.io.
    """
    def ingest_pdf(pdf_path: str, extract_images: bool = True, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest a multimodal PDF document and extract structured content.
        
        Args:
            pdf_path (str): Path to the PDF file
            extract_images (bool): Whether to extract and save images
            output_dir (str, optional): Directory to save extracted images
            
        Returns:
            Dict[str, Any]: Dictionary containing extracted content
        """
        try:
            logger.info(f"Ingesting PDF: {pdf_path}")
            
            # Create output directory if specified and extract_images is True
            if extract_images and output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Partition the PDF
            elements = partition_pdf(
                filename=pdf_path,
                extract_images_in_pdf=extract_images,
                infer_table_structure=True,
                include_page_breaks=True
            )
            
            # Process the elements
            content = {
                "text": [],
                "titles": [],
                "tables": [],
                "images": [],
                "page_breaks": [],
                "metadata": {}
            }
            
            # Extract metadata if available
            if hasattr(elements, "metadata") and elements.metadata:
                content["metadata"] = elements.metadata
            
            # Process each element
            for i, element in enumerate(elements):
                # Track page breaks
                if hasattr(element, "page_number") and element.page_number:
                    content["page_breaks"].append({
                        "index": i,
                        "page_number": element.page_number
                    })
                
                # Process different element types
                if isinstance(element, Text):
                    content["text"].append({
                        "index": i,
                        "text": element.text,
                        "page_number": getattr(element, "page_number", None)
                    })
                
                if isinstance(element, Title):
                    content["titles"].append({
                        "index": i,
                        "text": element.text,
                        "page_number": getattr(element, "page_number", None)
                    })
                
                if isinstance(element, Table):
                    # Convert table to a structured format
                    table_data = []
                    if hasattr(element, "metadata") and element.metadata:
                        table_data = element.metadata.get("text_as_html", "")
                    
                    content["tables"].append({
                        "index": i,
                        "data": table_data,
                        "page_number": getattr(element, "page_number", None)
                    })
                
                if isinstance(element, Image):
                    image_info = {
                        "index": i,
                        "page_number": getattr(element, "page_number", None)
                    }
                    
                    # Save image if requested
                    if extract_images and output_dir and hasattr(element, "metadata"):
                        image_path = element.metadata.get("image_path")
                        if image_path and os.path.exists(image_path):
                            # Copy image to output directory
                            import shutil
                            image_filename = f"image_{i}.png"
                            output_path = os.path.join(output_dir, image_filename)
                            shutil.copy(image_path, output_path)
                            image_info["path"] = output_path
                    
                    content["images"].append(image_info)
            
            logger.info(f"Successfully ingested PDF with {len(elements)} elements")
            return content
            
        except Exception as e:
            logger.error(f"Error ingesting PDF: {str(e)}")
            return {
                "error": f"Error ingesting PDF: {str(e)}",
                "text": [],
                "titles": [],
                "tables": [],
                "images": [],
                "page_breaks": []
            }
    
    # Create a LangChain tool for PDF ingestion
    pdf_ingestion_tool = Tool(
        name="pdf_ingestion",
        description="Useful for ingesting multimodal PDF documents and extracting structured content including text, titles, tables, and images. Input should be a path to a PDF file.",
        func=ingest_pdf
    )
    
    return pdf_ingestion_tool

# Example usage
if __name__ == "__main__":
    # Initialize the PDF ingestion tool
    pdf_tool = setup_unstructured_pdf_ingestion()
    
    # Example: Ingest a PDF
    pdf_path = "path/to/your/document.pdf"  # Replace with your PDF path
    output_dir = "extracted_content"  # Optional: directory to save images
    
    result = pdf_tool.run(pdf_path, extract_images=True, output_dir=output_dir)
    
    # Print summary of extracted content
    print(f"Extracted {len(result['text'])} text elements")
    print(f"Extracted {len(result['titles'])} titles")
    print(f"Extracted {len(result['tables'])} tables")
    print(f"Extracted {len(result['images'])} images")
    
    # Print first few text elements
    print("\nSample text elements:")
    for i, text_elem in enumerate(result["text"][:3]):
        print(f"{i+1}. {text_elem['text'][:100]}...")
    
    # Print image paths if available
    if result["images"]:
        print("\nExtracted images:")
        for img in result["images"]:
            if "path" in img:
                print(f"- {img['path']}") 