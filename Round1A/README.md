# Adobe Hackathon - Round 1A

## Overview
Round 1A focuses on extracting structured outlines from PDF documents using semantic analysis. The system automatically detects headings and titles within PDFs to create organized outlines.

## Features
- Automatic detection of headings and titles in PDF documents
- Semantic analysis to identify and extract meaningful sections
- Generation of structured JSON output containing document outlines
- Support for multiple PDF documents in the input directory

## Directory Structure
```
Round1A/
├── input/          # Directory for PDF input files
├── output/         # Directory for JSON output files
├── main.py         # Main script for PDF outline extraction
└── requirements.txt # Python dependencies
```

## Setup
1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Place PDF files in the `input/` directory

## Usage
1. Run the main script:
```bash
python main.py
```

2. Output files will be generated in the `output/` directory with the following format:
- Each PDF will generate a corresponding JSON file with the same name
- The JSON files contain structured outlines of the PDF documents

## Output Format
The output JSON files contain:
- Document title and metadata
- Extracted sections with titles and page numbers
- Semantic analysis of section importance
- Hierarchical structure of the document

## Dependencies
- PyMuPDF (fitz) - For PDF text extraction
- Python standard libraries

## Note
The system is designed to handle various PDF structures and formats, automatically detecting headings based on font size, style, and content context.