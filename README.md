# 🧠 AI Text Analyzer API

> Built with FastAPI — analyze, detect, and transform text intelligently.

This API provides various NLP (Natural Language Processing) utilities such as:

- Sentiment analysis
- Language detection
- Translation
- Text correction
- Emotion (tone) detection
- Word, sentence, and character count
- Text similarity
- Summarization
- Keyword extraction

---

## 🚀 Features

| Endpoint                    | Description                                      |
| --------------------------- | ------------------------------------------------ |
| `POST /sentiment`           | Analyze text sentiment (polarity & subjectivity) |
| `POST /language`            | Detect language using `langid`                   |
| `POST /available-languages` | Get all supported translation languages          |
| `POST /translate`           | Translate text into any target language          |
| `POST /correct`             | Auto-correct spelling in text                    |
| `POST /detect-tone`         | Detect emotional tone (happy, sad, angry, etc.)  |
| `POST /wordcount`           | Count words, sentences, and characters           |
| `POST /similarity`          | Compare similarity between two texts             |
| `POST /summarize`           | Summarize long text using LSA algorithm          |
| `POST /keywords`            | Extract top keywords with relevance scores       |

---

## 🧩 Installation

1. Clone this repo:
   ```bash
   git clone https://github.com/Chrisdiantodoni/ai-text-analyzer-api.git
   cd ai-text-analyzer-api
   ```
