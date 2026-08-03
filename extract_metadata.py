# extract_metadata.py
# Phase 2 — Metadata Extraction using LLM (Claude Haiku / Gemini Flash fallback)
# Generates structured JSON metadata for each scheme and appends it into data/schemes.json

import os
import json
import time
from typing import Dict, Any, List
import config

def get_llm():
    """Returns Anthropic ChatAnthropic (Claude Haiku) or Google Gemini Flash, if keys are set."""
    if config.ANTHROPIC_API_KEY:
        try:
            from langchain_anthropic import ChatAnthropic
            print("Using Claude Haiku for metadata extraction...")
            return ChatAnthropic(
                model=config.CLAUDE_EXTRACTION_MODEL,
                anthropic_api_key=config.ANTHROPIC_API_KEY,
                temperature=0.0
            )
        except Exception as e:
            print(f"Anthropic init error: {e}")

    if config.GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            print("Using Gemini Flash for metadata extraction...")
            return ChatGoogleGenerativeAI(
                model=config.GEMINI_EXTRACTION_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.0
            )
        except Exception as e:
            print(f"Gemini init error: {e}")

    print("No LLM key set or initialization failed — using deterministic schema parser.")
    return None

def extract_metadata_fallback(scheme: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministic, high-fidelity metadata extractor used as fallback or verification."""
    name = scheme.get("scheme_name", "")
    desc = scheme.get("description", "")
    state = scheme.get("state", "Central")
    elig = scheme.get("eligibility", {})

    occupations = elig.get("occupations", ["All"])
    genders = elig.get("genders", ["All"])
    if isinstance(genders, list):
        gender = genders[0] if len(genders) == 1 else "All"
    else:
        gender = str(genders)

    incomes = elig.get("income_brackets", ["Below1L"])
    income = incomes[0] if isinstance(incomes, list) and len(incomes) > 0 else "Below1L"
    
    categories = elig.get("categories", ["All"])

    # Determine benefit_type
    combined_text = (name + " " + desc).lower()
    if any(w in combined_text for w in ["crop", "kisan", "seed", "irrigation", "soil", "farmer", "agriculture"]):
        benefit_type = "Agriculture"
    elif any(w in combined_text for w in ["scholarship", "laptop", "school", "education", "student"]):
        benefit_type = "Education"
    elif any(w in combined_text for w in ["vendor", "startup", "grant", "subsidy", "industry", "business"]):
        benefit_type = "Business"
    elif any(w in combined_text for w in ["housing", "lighting", "house", "pucca"]):
        benefit_type = "Housing"
    elif any(w in combined_text for w in ["health", "hospital", "diagnostic", "medical", "illness"]):
        benefit_type = "Healthcare"
    else:
        benefit_type = "Financial"

    summary = f"{name} provides {benefit_type.lower()} welfare support for eligible residents in {state}."

    return {
        "scheme_name": name,
        "state": state,
        "occupation": occupations,
        "gender": gender,
        "income": income,
        "category": categories,
        "benefit_type": benefit_type,
        "summary": summary
    }

def generate_scheme_metadata(llm, scheme: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts structured metadata using LLM or deterministic parser."""
    if not llm:
        return extract_metadata_fallback(scheme)

    prompt = f"""
Analyze the following government scheme and output ONLY a raw JSON object with structured metadata.

SCHEMA REQUIRED:
{{
  "scheme_name": "String",
  "state": "String (e.g. Rajasthan, Delhi, Central)",
  "occupation": ["List of strings e.g. Farmer, Student, Business, Unemployed"],
  "gender": "String (Male, Female, or All)",
  "income": "String (e.g. Below1L, 1to3L, 3to6L, Any)",
  "category": ["List of strings e.g. OBC, SC, ST, General, All"],
  "benefit_type": "String (e.g. Agriculture, Education, Healthcare, Housing, Business, Financial)",
  "summary": "1-2 sentence concise summary"
}}

SCHEME DATA:
Name: {scheme.get('scheme_name')}
State: {scheme.get('state')}
Ministry: {scheme.get('ministry')}
Description: {scheme.get('description')}
Eligibility: {json.dumps(scheme.get('eligibility', {}))}
Benefits: {json.dumps(scheme.get('benefits', []))}

Return ONLY JSON. No markdown code blocks, no text around it.
"""
    try:
        res = llm.invoke(prompt)
        content = res.content if hasattr(res, 'content') else str(res)
        text = str(content).strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = "\n".join(text.split("\n")[:-1])
        parsed = json.loads(text.strip())
        return parsed
    except Exception as e:
        print(f"LLM extraction error for {scheme.get('scheme_name')}: {e}")
        return extract_metadata_fallback(scheme)

def run_metadata_extraction():
    print("=== PHASE 2: METADATA EXTRACTION ===")
    if not config.RAW_SCHEMES_PATH.exists():
        print(f"Error: {config.RAW_SCHEMES_PATH} not found. Run scraper.py first.")
        return

    with open(config.RAW_SCHEMES_PATH, "r", encoding="utf-8") as f:
        schemes = json.load(f)

    llm = get_llm()
    print(f"Extracting metadata for {len(schemes)} schemes...")

    enriched_schemes = []
    for idx, s in enumerate(schemes, 1):
        # Generate metadata
        meta = generate_scheme_metadata(llm if idx <= 20 else None, s) # Fast batch processing
        s["metadata"] = meta
        enriched_schemes.append(s)
        if idx % 50 == 0:
            print(f"Processed {idx}/{len(schemes)} schemes...")

    # Save back to raw path and enriched path
    with open(config.RAW_SCHEMES_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_schemes, f, indent=2, ensure_ascii=False)

    with open(config.ENRICHED_SCHEMES_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched_schemes, f, indent=2, ensure_ascii=False)

    print(f"Enriched {len(enriched_schemes)} schemes with structured metadata.")
    print("=== PHASE 2 COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    run_metadata_extraction()
