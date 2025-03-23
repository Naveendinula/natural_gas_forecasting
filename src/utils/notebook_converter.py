"""
Jupyter Notebook Converter

Utility to convert Jupyter notebooks to Python modules.
This can be used to convert the original etl_pipeline.ipynb to Python code.
"""

import argparse
import json
import re
import os
import logging

# Configure logging
logging.basicConfig(
    filename='notebook_converter.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_notebook_to_py(notebook_path, output_path=None, include_markdown=False, add_comments=True):
    """Convert a Jupyter notebook to a Python file.
    
    Args:
        notebook_path (str): Path to the notebook file.
        output_path (str, optional): Path to save the Python file.
        include_markdown (bool, optional): Whether to include markdown cells as comments.
        add_comments (bool, optional): Whether to add cell separator comments.
        
    Returns:
        str: Path to the created Python file.
    """
    try:
        logger.info(f"Converting notebook {notebook_path} to Python")
        
        # Determine output path if not provided
        if output_path is None:
            output_path = os.path.splitext(notebook_path)[0] + '.py'
            
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            
        # Initialize Python code
        py_code = [
            f"# Converted from {os.path.basename(notebook_path)}",
            "# This file was automatically generated from a Jupyter notebook.",
            ""
        ]
        
        # Extract cells
        cells = notebook.get('cells', [])
        
        # Process each cell
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type', '')
            source = cell.get('source', [])
            
            # Skip empty cells
            if not source:
                continue
                
            # Join source lines
            if isinstance(source, list):
                source = ''.join(source)
            
            # Process based on cell type
            if cell_type == 'code':
                if add_comments:
                    py_code.append(f"# Cell {i+1} (Code)")
                py_code.append(source)
                py_code.append("")  # Add blank line
            elif cell_type == 'markdown' and include_markdown:
                if add_comments:
                    py_code.append(f"# Cell {i+1} (Markdown)")
                # Convert markdown to comments
                markdown_lines = source.split('\n')
                commented_markdown = ['# ' + line for line in markdown_lines]
                py_code.append('\n'.join(commented_markdown))
                py_code.append("")  # Add blank line
        
        # Write Python code to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(py_code))
            
        logger.info(f"Successfully converted notebook to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error converting notebook: {e}")
        raise

def clean_py_file(py_file_path, output_path=None):
    """Clean a Python file generated from a notebook.
    
    Args:
        py_file_path (str): Path to the Python file to clean.
        output_path (str, optional): Path to save the cleaned file.
        
    Returns:
        str: Path to the cleaned Python file.
    """
    try:
        logger.info(f"Cleaning Python file {py_file_path}")
        
        # Determine output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(py_file_path)
            output_path = f"{base}_clean{ext}"
            
        # Read the Python file
        with open(py_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Remove magic commands
        content = re.sub(r'^%.*$', '', content, flags=re.MULTILINE)
        content = re.sub(r'^!.*$', '', content, flags=re.MULTILINE)
        
        # Remove display commands
        content = re.sub(r'^display\(.*\)$', '', content, flags=re.MULTILINE)
        
        # Remove consecutive blank lines
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Write cleaned content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        logger.info(f"Successfully cleaned Python file to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error cleaning Python file: {e}")
        raise

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Convert Jupyter notebooks to Python files")
    
    parser.add_argument('notebook_path', help="Path to the notebook file")
    parser.add_argument('--output', '-o', help="Path to save the Python file")
    parser.add_argument('--include-markdown', '-m', action='store_true', 
                        help="Include markdown cells as comments")
    parser.add_argument('--no-comments', '-n', action='store_true',
                        help="Don't add cell separator comments")
    parser.add_argument('--clean', '-c', action='store_true',
                        help="Clean the generated Python file")
    
    args = parser.parse_args()
    
    try:
        # Convert notebook to Python
        py_file = convert_notebook_to_py(
            args.notebook_path,
            args.output,
            args.include_markdown,
            not args.no_comments
        )
        
        # Clean the Python file if requested
        if args.clean:
            clean_py_file(py_file)
            
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == "__main__":
    main() 