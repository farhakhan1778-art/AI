🏛️ Yojana Sahayak (योजना सहायक)
Standard RAG AI Chatbot for Indian Government Scheme Discovery
Yojana Sahayak is a production-ready AI application designed to help Indian citizens discover relevant government welfare schemes based on their personal profile. It uses a Standard RAG (Retrieval-Augmented Generation) architecture built with ChromaDB, LangChain, SentenceTransformers, and Streamlit.

🌟 Key Features
💬 Dynamic 7-Step Questionnaire: Asks ONLY ONE question at a time to prevent user fatigue (State/UT, Age, Gender, Occupation, Annual Income, Social Category, and Specific Interest).
📊 Real-Time Live Match Counter: Displays an interactive counter ("Found XX matching schemes so far") that updates dynamically as the user answers questions.
⚡ ChromaDB Metadata Pre-Filtering: Filters vector search space by State (e.g., Rajasthan + Central), Gender, Occupation, Income, and Category before running semantic retrieval.
🧠 Dense Vector Embeddings: Uses sentence-transformers/all-MiniLM-L6-v2 (384 dimensions) for deep semantic matching against 250+ structured government schemes.
🤖 LLM Scheme Ranking & Recommendations: Ranks schemes using Claude 3.5 Sonnet (or Google Gemini Flash fallback) and formats clear recommendation cards highlighting eligibility reasons, specific benefits, application steps, and official portal links.
🎨 Premium Streamlit UI: Dark-mode glassmorphic split-panel layout featuring profile completion tracking, citizen profile card, chat timeline with avatar bubbles, quick-select option buttons, and rank badges (#1 MATCH, #2 MATCH).

🛠️ Tech Stack
Component	Technology / Library
Web Scraping & Data Collection	requests, beautifulsoup4 (Scrapes myscheme.gov.in & builds 250+ scheme dataset)
Metadata Extraction	LLM Structured JSON Extraction (Claude Haiku / Gemini Flash fallback)
Vector Database	ChromaDB (Persistent client stored at ./chroma_db)
Embedding Model	sentence-transformers/all-MiniLM-L6-v2 (Local, 384-dimensional dense vectors)
RAG Framework	LangChain (langchain-anthropic, langchain-google-genai, langchain-community)
Frontend UI	Streamlit
Environment & Config	python-dotenv, pydantic
Language	Python 3.11+
📐 System Architecture
Mermaid diagram
📁 Repository Structure
text

AI launchpad/
│
├── 📄 app.py                  # Streamlit frontend UI (split-panel layout, chat stream, live counter badge)
├── 📄 chatbot.py              # Turn-by-turn questionnaire manager & user profile state controller
├── 📄 retrieval_pipeline.py    # LangChain RAG pipeline (ChromaDB search, metadata filter, LLM ranking)
├── 📄 vector_store.py         # ChromaDB persistent store & sentence-transformers embedding manager
├── 📄 extract_metadata.py     # Phase 2: Metadata extraction script (structured JSON enrichment)
├── 📄 scraper.py              # Phase 1: Web scraper & 250+ government scheme dataset builder
├── 📄 config.py               # Central configuration loader & environment constants
├── 📄 verify_pipeline.py      # Automated end-to-end verification test script
├── 📄 create_demo_video.py     # 1080p MP4 video demonstration renderer
│
├── 📂 data/                   # Data directory
│   ├── 📄 schemes.json                 # Raw structured JSON dataset (250+ schemes)
│   ├── 📄 schemes_with_metadata.json   # Enriched JSON dataset with extracted metadata
│   └── 📂 schemes_txt/                 # Cleaned text versions for embedding generation
│
├── 📂 chroma_db/              # Persistent ChromaDB vector database index
│
├── 📄 requirements.txt        # Python package dependencies
├── 📄 .env.example            # Environment variables template
└── 📄 README.md               # Project documentation
🚀 Quick Start Guide
1. Prerequisites
Ensure Python 3.11+ is installed. Clone or open the project directory:

bash

cd "AI launchpad"
2. Environment Setup
Create a .env file in the root directory (refer to .env.example):

env

# API Credentials (Anthropic Claude or Google Gemini)
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GOOGLE_API_KEY=your-google-api-key-here
# Model Names
CLAUDE_CHAT_MODEL=claude-3-5-sonnet-20241022
CLAUDE_EXTRACTION_MODEL=claude-3-5-haiku-20241022
3. Install Dependencies
bash

pip install -r requirements.txt
4. Build Dataset & Vector Database
Run the pipeline scripts in sequence:

bash

# Phase 1: Scrape & build 250+ schemes dataset
python scraper.py
# Phase 2: Extract structured JSON metadata
python extract_metadata.py
# Phase 3: Build ChromaDB vector store
python vector_store.py
5. Launch the Streamlit Web App
bash

streamlit run app.py
Open http://localhost:8501 in your web browser to interact with Yojana Sahayak.

🧪 Verification & Testing
Run the end-to-end automated verification script to test the 7-step questionnaire flow and RAG pipeline:

bash

python verify_pipeline.py
📜 License
This project is open-source and available under the MIT License.
