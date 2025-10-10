from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from textblob import TextBlob
import langid
from deep_translator import GoogleTranslator
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import yake
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import text2emotion as te


app = FastAPI(title="AI Text Analyzer API")

class TextRequest(BaseModel):
    text: str

class CompareRequest(BaseModel):
    text1: str
    text2: str


@app.post("/sentiment")
def analyze_sentiment(request: TextRequest):
    """
    Analyzes the sentiment of the provided text.
    """
    blob = TextBlob(request.text)
    return {
        "polarity": blob.sentiment.polarity, # pyright: ignore[reportAttributeAccessIssue]
        "subjectivity": blob.sentiment.subjectivity # pyright: ignore[reportAttributeAccessIssue]
    }

@app.post("/language")
def detect_language(request: TextRequest):
    """
    Detects the language of the provided text using langid.py.
    """
    lang_code, confidence = langid.classify(request.text)
    if confidence < 0.85 and len(request.text.split()) < 3:
        lang_code = "unknown"
    return {"language": lang_code, "confidence": confidence}

@app.post("/available-languages")
def get_all_available_languages():
    """
    Get all available languages
    """
    try:
        languages = GoogleTranslator().get_supported_languages(as_dict=True)
        return {"languages": languages}
        pass
    except Exception as e:
        raise e

@app.post("/translate")
def translate_text(request: TextRequest, target_lang:str = Query("en", description="Target language code, e.g. 'en', 'id', 'fr'")):
    """
    Translates the provided text to any language within lists using GoogleTranslator from deep_translator.
    """
    try:
        # Use GoogleTranslator from deep_translator
        translated_text = GoogleTranslator(source='auto', target=target_lang).translate(request.text)
        if not translated_text:
            raise HTTPException(status_code=500, detail="Translation failed, returned empty text.")
        return {
            "original_text": request.text,
            "translated_text": translated_text,
            "target_language": target_lang
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/correct")
def correct_text(request: TextRequest):
    blob = TextBlob(request.text)
    corrected = str(blob.correct())
    return {"original": request.text, "corrected": corrected}


@app.post("/detect-tone")
def detect_tone(request: TextRequest):
    """
    Detects the emotional tone of the provided text.
    Returns a distribution of emotions.
    """
    try:
        emotions = te.get_emotion(request.text)
        # Example output: {'Happy': 0.5, 'Angry': 0.0, 'Surprise': 0.0, 'Sad': 0.5, 'Fear': 0.0}
        dominant_emotion = max(emotions, key=emotions.get)
        return {
            "text": request.text,
            "dominant_emotion": dominant_emotion,
            "scores": emotions
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tone detection failed: {str(e)}")


@app.post("/wordcount")
def word_count(request: TextRequest):
    text = request.text
    words = len(re.findall(r'\w+', text))
    sentences = text.count('.') + text.count('!') + text.count('?')
    chars = len(text)
    return {
        "words": words,
        "sentences": sentences,
        "characters": chars
    }

@app.post("/similarity")
def compare_texts(request: CompareRequest):
    vectorizer = TfidfVectorizer().fit_transform([request.text1, request.text2])
    sim_score = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    return {"similarity": round(float(sim_score), 3)}


@app.post("/summarize")
def summarize_text(request: TextRequest):
    parser = PlaintextParser.from_string(request.text, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 3)  # ambil 3 kalimat
    return {"summary": " ".join(str(sentence) for sentence in summary)}

@app.post("/keywords")
def extract_keywords(request: TextRequest):
    """
    Extracts top keywords and their scores using YAKE.
    Lower score = higher relevance.
    """
    kw_extractor = yake.KeywordExtractor(top=10)
    keywords = kw_extractor.extract_keywords(request.text)
    return {
        "keywords": [
            {"keyword": kw, "score": score}
            for kw, score in keywords
        ]
    }
