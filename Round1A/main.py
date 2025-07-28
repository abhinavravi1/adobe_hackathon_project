import fitz  # PyMuPDF
import os
import json
import langdetect
from langdetect import detect_langs

def analyze_language(text):
    """Detect language and return confidence score"""
    try:
        langs = detect_langs(text)
        # Return the most probable language with its confidence score
        return langs[0].lang, langs[0].prob
    except:
        return "unknown", 0.0

def extract_outline(pdf_path):
    doc = fitz.open(pdf_path)
    outline = []
    
    # Try to get a better title from the first page
    first_page = doc[0]
    first_page_text = first_page.get_text("text").strip()
    if first_page_text:
        # Split by newlines and take the first non-empty line as title
        lines = first_page_text.split('\n')
        title = next((line.strip() for line in lines if line.strip()), os.path.basename(pdf_path).replace(".pdf", ""))
    else:
        title = os.path.basename(pdf_path).replace(".pdf", "")

    # Analyze font sizes and styles
    font_info = {}
    text_blocks = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    size = round(span["size"], 1)
                    text = span["text"].strip()
                    if not text:
                        continue
                    
                    # Store text blocks for later analysis
                    text_blocks.append({
                        "text": text,
                        "size": size,
                        "font": span["font"],
                        "page": page_num + 1
                    })
                    
                    key = (size, span["font"])
                    font_info[key] = font_info.get(key, 0) + 1

    # Analyze text blocks to find headings
    structured = []
    for block in text_blocks:
        text = block["text"]
        size = block["size"]
        page = block["page"]
        
        # Check if this looks like a heading based on:
        # 1. Text starts with a number (e.g., "1.", "2.1")
        # 2. Text is all uppercase
        # 3. Text ends with a colon
        # 4. Text is significantly larger than surrounding text
        is_heading = False
        
        if text and any(text.startswith(str(i) + ".") for i in range(1, 10)):
            is_heading = True
        elif text.isupper():
            is_heading = True
        elif text.strip().endswith(":"):
            is_heading = True
        
        if is_heading:
            structured.append({
                "level": "H1",
                "text": text,
                "page": page
            })
        elif size > 12:  # Consider larger text as potential heading
            structured.append({
                "level": "H2",
                "text": text,
                "page": page
            })

    # Remove duplicate headings
    seen_texts = set()
    final_structured = []
    for item in structured:
        if item["text"] not in seen_texts:
            seen_texts.add(item["text"])
            final_structured.append(item)

    # Add language information to each heading
    for item in final_structured:
        lang, confidence = analyze_language(item["text"])
        item["language"] = lang
        item["language_confidence"] = confidence

    return {
        "title": title,
        "outline": final_structured
    }

    return {
        "title": title,
        "outline": structured
    }

def main():
    # Get the absolute path to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "input")
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(input_dir, filename)
            output = extract_outline(filepath)
            output_path = os.path.join(output_dir, filename.replace(".pdf", ".json"))
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()