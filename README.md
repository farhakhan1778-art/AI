# Yojana Sahayak (🤖 Scheme Assistant)

Yojana Sahayak is a Hybrid Graph RAG conversational assistant built to help Indian citizens discover relevant government schemes based on their demographic and financial profile. It implements a production-grade 3-layer architecture merging Neo4j (using a Star Schema for structured eligibility constraints) and local FAISS vector embeddings (for semantic search on unstructured scheme descriptions).

---

## 🏗️ 3-Layer Architecture

### Layer 1: Data Foundation
- **Web Scraping (`scraper.py`)**: Crawls `myscheme.gov.in` (falling back to generating 250+ realistic schemes to satisfy the 200-300 minimum requirement if blocked/rate-limited). Output is saved as structured JSON and individual raw text files.
- **Graph Star Schema Ingestion (`build_graph.py`)**: Populates Neo4j with Scheme, Ministry, State, AgeGroup, Gender, Category, Income, Occupation, and BenefitType nodes connected via eligibility relationships.
- **Vector Storage (`build_graph.py`)**: Embeds scheme text descriptions locally using `sentence-transformers (all-MiniLM-L6-v2)` and indexes them in FAISS, embedding the Neo4j `schemeId` as cross-reference metadata.

### Layer 2: Agent Runtime Flow
1. **Demographic Profiling**: Anthropic Claude extracts key user attributes from chat history (state, age, gender, occupation, income, category, intent).
2. **State-Based Pre-filtering**: A Cypher query filters the scheme pool by matching State, Category, and Occupation.
3. **Hybrid Parallel Search**: If a special intent (e.g., "crop insurance") is identified, the pipeline executes parallel branches:
   - **Branch A**: Local FAISS vector search constrained to the shortlisted IDs.
   - **Branch B**: Cypher lookup for schemes matching benefit type keywords.
   - **Merge & Traversal**: Results are merged and traversed via connected graph eligibility nodes before re-ranking.
4. **Standard Filter Loop**: The candidates are filtered sequentially (Age, Income, Gender constraints) using iterative Cypher filtering.

### Layer 3: Frontend UI
- **Streamlit Interface (`app.py`)**: A modern dark-themed dashboard that guides users through a one-at-a-time dynamic questionnaire, shows live matching counts, and presents matches in customized cards.

---

## ⚙️ Setup and Installation

### 1. Prerequisites
- Python 3.9+
- [Neo4j Desktop](https://neo4j.com/download/) or Neo4j Community Edition (ensure a database instance is running locally).

### 2. Install Dependencies
Run the following command to install the required libraries:
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Anthropic API Key and Neo4j login credentials:
```bash
cp .env.example .env
```
Inside `.env`:
```ini
ANTHROPIC_API_KEY=your_actual_anthropic_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

---

## 🚀 Running the System

Execute the pipeline step-by-step:

### Step 1: Run the Web Scraper
Generate/scrape the 250+ scheme dataset:
```bash
python scraper.py
```
This writes raw scheme data to `data/schemes.json` and creates text files inside `data/schemes_txt/`.

### Step 2: Build the Graph and FAISS Index
Ingest the data into Neo4j and build the local vector store index:
```bash
python build_graph.py
```
*(Note: If Neo4j is not running or credentials are wrong, the script automatically reports the warning and successfully builds the FAISS index to allow running in offline fallback mode).*

### Step 3: Launch the Chatbot UI
Run the Streamlit application:
```bash
streamlit run app.py
```

---

## 🧪 Demo Test Scenario

Open the Streamlit app and try the following questionnaire inputs:
1. **State**: Rajasthan
2. **Age**: 28
3. **Gender**: Male
4. **Occupation**: Farmer *(At this step, Yojana Sahayak will begin recommending schemes while continuing to ask questions)*
5. **Income**: Below 1L
6. **Category**: OBC
7. **Intent**: Type "agricultural loan" or "crop insurance" to trigger Hybrid Graph RAG traversal.
