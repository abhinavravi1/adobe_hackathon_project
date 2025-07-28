import os
import json
from sentence_transformers import SentenceTransformer, util
import fitz
import langdetect
from langdetect import detect_langs
import re
from collections import Counter
import numpy as np
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import time
from tqdm import tqdm

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_text_by_page(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text().strip()
        if text:
            # Split text into sentences
            sentences = sent_tokenize(text)
            
            # Get font information
            blocks = doc[i].get_text("dict")["blocks"]
            font_sizes = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(round(span["size"], 1))
            
            avg_font_size = np.mean(font_sizes) if font_sizes else 12
            
            pages.append({
                "page": i + 1,
                "text": text,
                "sentences": sentences,
                "avg_font_size": avg_font_size,
                "num_sentences": len(sentences)
            })
    return pages

def analyze_language(text):
    """Detect language and return confidence score"""
    try:
        # Handle dict input by extracting text content
        if isinstance(text, dict):
            text = text.get('text', '')
        
        # Handle None or empty input
        if not text:
            return {"unknown": 1.0}
            
        # Ensure text is a string
        text = str(text)
        
        # Clean text by removing special characters
        text = re.sub(r'[\W_]+', ' ', text)
        
        # Skip if text is too short for reliable detection
        if len(text.split()) < 5:
            return {"unknown": 1.0}
            
        langs = detect_langs(text)
        # Get top 3 languages with their probabilities
        lang_probs = {lang.lang: lang.prob for lang in langs[:3]}
        
        # If no confident detection, return unknown
        if max(lang_probs.values()) < 0.7:
            return {"unknown": 1.0}
            
        return lang_probs
    except Exception as e:
        print(f"Language detection error: {str(e)}")
        return {"unknown": 1.0}

def extract_keywords(text, persona_keywords):
    """Extract relevant keywords from text based on persona keywords"""
    # Initialize NLP tools
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Get text content from dict if needed
    if isinstance(text, dict):
        text = text.get('text', '')
    
    # Tokenize and preprocess text
    if not text:
        return {}
    
    # Clean text by removing special characters and numbers
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    clean_text = re.sub(r'\d+', '', clean_text)
    
    # Extract words
    words = re.findall(r'\w+', clean_text)
    
    # Apply lemmatization and filtering
    processed_words = []
    for word in words:
        # Skip common stop words
        if word in stop_words:
            continue
            
        # Skip very short words
        if len(word) <= 2:
            continue
            
        # Lemmatize
        lemma = lemmatizer.lemmatize(word)
        
        # Skip if lemmatization didn't change the word
        if lemma == word:
            continue
            
        processed_words.append(lemma)
    
    # Create keyword variations
    keyword_variations = set()
    for keyword in persona_keywords:
        # Add base keyword
        keyword_variations.add(keyword.lower())
        
        # Add lemmatized version
        lemma = lemmatizer.lemmatize(keyword)
        keyword_variations.add(lemma)
        
        # Add common variations
        if keyword.endswith('e'):
            keyword_variations.add(keyword[:-1])
        if keyword.endswith('s'):
            keyword_variations.add(keyword[:-1])
            
        # Add plural/singular forms
        if keyword.endswith('y'):
            keyword_variations.add(keyword[:-1] + 'ies')
            keyword_variations.add(keyword[:-1] + 'y')
    
    # Count keyword matches
    keyword_counts = Counter(w for w in processed_words if w in keyword_variations)
    
    # Add context-aware variations
    context_keywords = set()
    for keyword in keyword_counts:
        # Add related words (simple variations)
        if keyword.endswith('ing'):
            context_keywords.add(keyword[:-3])
        if keyword.endswith('ed'):
            context_keywords.add(keyword[:-2])
            context_keywords.add(keyword[:-2] + 'ing')
        
    # Update counts with context keywords
    for keyword in context_keywords:
        if keyword in processed_words:
            keyword_counts[keyword] += 1
    
    return keyword_counts
    
    # Extract keywords with variations
    keyword_variations = set()
    for keyword in persona_keywords:
        # Add base keyword
        keyword_variations.add(keyword)
        # Add lemmatized version
        keyword_variations.add(lemmatizer.lemmatize(keyword))
        # Add common variations
        if keyword.endswith('e'):
            keyword_variations.add(keyword[:-1])
        if keyword.endswith('s'):
            keyword_variations.add(keyword[:-1])
    
    keyword_counts = Counter(w for w in words if w in keyword_variations)
    return keyword_counts

def calculate_relevance_score(text, persona_task_embedding, persona_keywords):
    """Calculate comprehensive relevance score with enhanced features"""
    # Ensure text is a string
    if isinstance(text, dict):
        text_content = text.get('text', '')
    else:
        text_content = str(text)
    
    # Semantic similarity score
    emb = model.encode(text_content, convert_to_tensor=True)
    semantic_score = util.pytorch_cos_sim(emb, persona_task_embedding)[0][0].item()
    
    # Keyword matching score with variations
    keywords = extract_keywords(text, persona_keywords)
    keyword_score = min(len(keywords) / len(persona_keywords), 1.0)
    
    # Language confidence
    lang_probs = analyze_language(text)
    lang_confidence = max(lang_probs.values())
    
    # Text quality features
    if isinstance(text, dict):
        text_content = text.get('text', '')
    else:
        text_content = str(text)
    
    if not text_content:
        num_sentences = 0
        avg_sentence_length = 0
    else:
        sentences = sent_tokenize(text_content)
        num_sentences = len(sentences)
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
    
    # Font size normalization
    avg_font_size = 12  # Default if not provided
    if isinstance(text, dict):  # If text is a page dict
        avg_font_size = text.get('avg_font_size', 12)
    
    # Combine features with weights
    weights = {
        'semantic': 0.4,  # Reduced weight
        'keywords': 0.3,
        'language': 0.1,
        'text_quality': 0.1,
        'font_size': 0.1
    }
    
    # Calculate weighted score
    score = (
        weights['semantic'] * semantic_score +
        weights['keywords'] * keyword_score +
        weights['language'] * lang_confidence +
        weights['text_quality'] * (num_sentences * avg_sentence_length) +
        weights['font_size'] * (avg_font_size / 12)  # Normalize font size
    )
    
    return score, lang_probs, keywords, {
        'num_sentences': num_sentences,
        'avg_sentence_length': avg_sentence_length,
        'avg_font_size': avg_font_size
    }

def rank_sections(pages, persona_task_embedding, pdf_path, persona_keywords):
    results = []
    for page in tqdm(pages, desc="Ranking sections"):
        start_time = time.time()
        
        text = page["text"]
        score, lang_probs, keywords, text_features = calculate_relevance_score(
            page, persona_task_embedding, persona_keywords
        )
        
        # Skip sections with very low relevance
        if score < 0.2:  # Threshold for minimum relevance
            continue
            
        results.append({
            "document": os.path.basename(pdf_path),
            "page_number": page["page"],
            "section_title": text[:60],
            "importance_rank": score,
            "language": lang_probs,
            "keywords": list(keywords.keys()),
            "confidence": score,
            "text_features": text_features,
            "processing_time": time.time() - start_time
        })
    
    # Sort by importance rank and take top 5
    sorted_results = sorted(results, key=lambda x: x["importance_rank"], reverse=True)
    return sorted_results[:5]

def process_collection(collection_dir):
    """Process a single collection of PDFs"""
    input_path = os.path.join(collection_dir, "challenge1b_input.json")
    output_path = os.path.join(collection_dir, "challenge1b_output.json")
    
    # Read input configuration
    with open(input_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Process each document
    extracted_sections = []
    subsection_analysis = []
    
    for doc in config["documents"]:
        pdf_path = os.path.join(collection_dir, "PDFs", doc["filename"])
        pages = extract_text_by_page(pdf_path)
        
        # Create persona task string and extract persona keywords
        persona_task = f"{config['persona']['role']} needs to {config['job_to_be_done']['task']}"
        base_keywords = set(re.findall(r'\w+', persona_task.lower()))
        
        # Safely get persona skills and interests
        persona_skills = config['persona'].get('skills', [])
        persona_interests = config['persona'].get('interests', [])
        
        persona_keywords = set(
            list(base_keywords) +
            (persona_skills if isinstance(persona_skills, list) else []) +
            (persona_interests if isinstance(persona_interests, list) else [])
        )
        
        persona_task_embedding = model.encode(persona_task, convert_to_tensor=True)
        
        # Rank sections
        ranked_sections = rank_sections(
            pages, persona_task_embedding, pdf_path, persona_keywords
        )
        
        # Add to results
        for section in ranked_sections:
            extracted_sections.append({
                "document": doc["filename"],
                "section_title": section["section_title"],
                "importance_rank": section["importance_rank"],
                "page_number": section["page_number"],
                "language": section["language"],
                "keywords": section["keywords"],
                "confidence": section["confidence"],
                "text_features": section["text_features"],
                "processing_time": section["processing_time"]
            })
            
            # Add subsection analysis with enhanced info
            page_text = next(p["text"] for p in pages if p["page"] == section["page_number"])
            subsection_analysis.append({
                "document": doc["filename"],
                "refined_text": page_text[:1000],  # Limit to first 1000 chars
                "page_number": section["page_number"],
                "language": section["language"],
                "keywords": section["keywords"],
                "text_features": section["text_features"],
                "processing_time": section["processing_time"]
            })
    
    # Create output
    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in config["documents"]],
            "persona": config['persona']['role'],
            "job_to_be_done": config['job_to_be_done']['task']
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    
    # Write output
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

def main():
    # Process each collection directory
    collections = ["Collection 1", "Collection 2", "Collection 3"]
    
    for collection in collections:
        collection_dir = os.path.join(os.path.dirname(__file__), collection)
        if os.path.exists(collection_dir):
            print(f"Processing collection: {collection}")
            process_collection(collection_dir)

if __name__ == "__main__":
    main()