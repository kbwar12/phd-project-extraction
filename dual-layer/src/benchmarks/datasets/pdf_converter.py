import os
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class PDFToMarkdownConverter:
    """
    Converts PDF documents to markdown format for better text processing
    """
    
    def __init__(self, pdf_dir: str, markdown_dir: str):
        self.pdf_dir = Path(pdf_dir)
        self.markdown_dir = Path(markdown_dir)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
    
    def convert_pdf_to_markdown(self, pdf_path: str) -> str:
        """
        Convert a single PDF file to markdown format
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Path to the generated markdown file
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create markdown filename
        markdown_filename = pdf_path.stem + ".md"
        markdown_path = self.markdown_dir / markdown_filename
        
        try:
            # Open PDF document
            doc = fitz.open(pdf_path)
            markdown_content = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Extract text with formatting
                text = page.get_text()
                
                # Basic markdown formatting
                if text.strip():
                    # Add page separator
                    if page_num > 0:
                        markdown_content.append(f"\n---\n**Page {page_num + 1}**\n")
                    
                    # Clean up text
                    cleaned_text = self._clean_text(text)
                    markdown_content.append(cleaned_text)
            
            doc.close()
            
            # Write markdown file
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(markdown_content))
            
            logger.info(f"Converted {pdf_path.name} to {markdown_path.name}")
            return str(markdown_path)
            
        except Exception as e:
            logger.error(f"Error converting {pdf_path}: {e}")
            raise
    
    def convert_all_pdfs(self) -> List[str]:
        """
        Convert all PDF files in the PDF directory to markdown
        
        Returns:
            List of paths to generated markdown files
        """
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        markdown_files = []
        
        logger.info(f"Found {len(pdf_files)} PDF files to convert")
        
        for pdf_file in pdf_files:
            try:
                markdown_path = self.convert_pdf_to_markdown(pdf_file)
                markdown_files.append(markdown_path)
            except Exception as e:
                logger.error(f"Failed to convert {pdf_file}: {e}")
                continue
        
        logger.info(f"Successfully converted {len(markdown_files)} PDF files")
        return markdown_files
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and format extracted text for better markdown structure
        """
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect headers (simple heuristic: short lines, often uppercase)
            if len(line) < 100 and line.isupper() and len(line.split()) <= 8:
                cleaned_lines.append(f"## {line.title()}")
            else:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def get_markdown_content(self, markdown_path: str) -> str:
        """
        Read markdown content from file
        
        Args:
            markdown_path: Path to markdown file
            
        Returns:
            Content of the markdown file
        """
        with open(markdown_path, 'r', encoding='utf-8') as f:
            return f.read()
