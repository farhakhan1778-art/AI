# retrieval_pipeline.py
# Phase 4 — RAG Retrieval Pipeline using LangChain, ChromaDB, and Claude Sonnet (with Gemini fallback)
# Executes vector retrieval, metadata filtering, and LLM scheme ranking/recommendation.

import json
from typing import Dict, Any, List, Optional
import config
from vector_store import SchemeVectorStore

# Load vector store singleton instance
_vector_store_instance = None

def get_vector_store() -> SchemeVectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = SchemeVectorStore()
    return _vector_store_instance

def get_chat_llm():
    """Returns Anthropic ChatAnthropic (Claude Sonnet) or Google Gemini Chat, if available."""
    if config.ANTHROPIC_API_KEY:
        try:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.CLAUDE_CHAT_MODEL,
                anthropic_api_key=config.ANTHROPIC_API_KEY,
                temperature=0.3
            )
        except Exception as e:
            print(f"Anthropic Chat LLM init error: {e}")

    if config.GOOGLE_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.GEMINI_CHAT_MODEL,
                google_api_key=config.GOOGLE_API_KEY,
                temperature=0.3
            )
        except Exception as e:
            print(f"Gemini Chat LLM init error: {e}")

    return None

def filter_candidates_by_profile(candidates: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Applies secondary in-memory eligibility filtering for age, income, category, and occupation."""
    filtered = []
    
    user_state = profile.get("state")
    user_age = profile.get("age")
    user_gender = profile.get("gender")
    user_occ = profile.get("occupation")
    user_inc = profile.get("income")
    user_cat = profile.get("category")

    for s in candidates:
        elig = s.get("eligibility", {})
        
        # 1. State filter
        if user_state and user_state.lower() not in ["all", "other", "central", "none"]:
            s_state = s.get("state", "central").lower()
            if s_state not in [user_state.lower(), "central"]:
                continue

        # 2. Age filter
        if user_age is not None and isinstance(user_age, (int, float)):
            age_min = elig.get("age_min", 0)
            age_max = elig.get("age_max", 100)
            if not (age_min <= user_age <= age_max):
                continue

        # 3. Gender filter
        if user_gender and user_gender.lower() in ["male", "female", "transgender"]:
            genders = [g.lower() for g in elig.get("genders", ["all"])]
            if user_gender.lower() not in genders and "all" not in genders:
                continue

        # 4. Occupation filter
        if user_occ and user_occ.lower() not in ["all", "other", "none"]:
            occs = [o.lower() for o in elig.get("occupations", ["all"])]
            if user_occ.lower() not in occs and "all" not in occs:
                continue

        # 5. Category filter
        if user_cat and user_cat.lower() not in ["all", "general", "other", "none"]:
            cats = [c.lower() for c in elig.get("categories", ["all"])]
            if user_cat.lower() not in cats and "all" not in cats and "general" not in cats:
                continue

        filtered.append(s)

    # Recovery: if strict filtering yields too few candidates, return candidates
    if len(filtered) < 2:
        return candidates[:5]
    return filtered

def format_recommendation_fallback(schemes: List[Dict[str, Any]], profile: Dict[str, Any]) -> str:
    """Fallback text generator when LLM is unavailable."""
    if not schemes:
        return "Abhi aapke exact profile ke liye target scheme nahi mili. Kripya myscheme.gov.in par check karein."

    lines = ["Namaste! Aapke profile ke mutabiq yeh best matching schemes hain:\n"]
    for idx, s in enumerate(schemes[:5], 1):
        lines.append(f"Scheme Name: {s.get('scheme_name')}")
        lines.append(f"Why you qualify: Matches your profile parameters ({profile.get('occupation','Citizen')}, {profile.get('state','India')}, {profile.get('category','All')})")
        lines.append(f"Benefit: {' '.join(s.get('benefits', []))[:140]}")
        lines.append(f"How to apply: {s.get('application_process', 'Apply online via portal or visit local CSC center.')}")
        lines.append(f"Link: {s.get('official_website', 'https://www.myscheme.gov.in')}")
        lines.append("")
    return "\n".join(lines)

def run_rag_pipeline(
    user_query: str,
    user_profile: Dict[str, Any],
    chat_history_str: str = ""
) -> Dict[str, Any]:
    """
    Standard RAG Retrieval Pipeline:
    1. Vector similarity search + ChromaDB metadata filtering (Top 10).
    2. In-memory profile eligibility validation.
    3. LLM (Claude Sonnet / Gemini) synthesis and ranking of top 3-5 schemes.
    """
    store = get_vector_store()
    
    # Formulate search intent text
    intent = user_profile.get("interest") or user_query or "welfare subsidy scheme"
    search_text = f"{intent} {user_profile.get('occupation', '')} {user_profile.get('state', '')}"

    # Step 2 & 3: ChromaDB Semantic Search + Metadata Filters
    raw_candidates = store.query_schemes(query=search_text, user_profile=user_profile, top_k=10)
    
    # In-memory validation
    candidates = filter_candidates_by_profile(raw_candidates, user_profile)
    top_schemes = candidates[:5]

    llm = get_chat_llm()
    if not llm or not top_schemes:
        response_text = format_recommendation_fallback(top_schemes, user_profile)
        return {
            "retrieved_schemes": top_schemes,
            "response": response_text
        }

    # Format scheme context for Claude Sonnet / Gemini
    scheme_context_parts = []
    for idx, s in enumerate(top_schemes, 1):
        scheme_context_parts.append(
            f"Scheme {idx}:\n"
            f"Name: {s.get('scheme_name')}\n"
            f"State: {s.get('state')}\n"
            f"Ministry: {s.get('ministry')}\n"
            f"Description: {s.get('description')}\n"
            f"Eligibility: {json.dumps(s.get('eligibility', {}))}\n"
            f"Benefits: {' '.join(s.get('benefits', []))}\n"
            f"Application Process: {s.get('application_process')}\n"
            f"Link: {s.get('official_website')}\n"
        )
    scheme_context = "\n".join(scheme_context_parts)

    system_prompt = """You are Yojana Sahayak, an AI assistant helping Indian citizens discover government welfare schemes.
Your personality is friendly, warm, helpful, and professional.
Use simple English with occasional natural Hindi words (Namaste, aapke liye, yojana, bahut accha).

CONVERSATION RULES:
- Ask only one question at a time.
- Recommend ONLY retrieved schemes provided in CONTEXT. Do not hallucinate schemes.
- Explain clearly why the user qualifies based on their profile.
- Describe specific benefits and clear application steps.
- Provide the official website link for each recommended scheme.

FORMAT REQUIREMENT:
For each recommended scheme (return 3 to 5 best matches), use EXACTLY this structure:

Scheme Name: [Exact scheme name]
Why you qualify: [Clear explanation matching user state, occupation, income, or category]
Benefit: [Specific benefits provided]
How to apply: [Application steps]
Link: [Official website URL]
"""

    user_prompt = f"""
USER PROFILE:
{json.dumps(user_profile, indent=2)}

RECENT CONVERSATION HISTORY:
{chat_history_str}

LATEST USER QUERY / ANSWER:
"{user_query}"

RETRIEVED SCHEMES CONTEXT:
{scheme_context}

INSTRUCTIONS:
1. Write a warm 1-line opening acknowledging their profile.
2. Recommend the best 3 to 5 matching schemes using the EXACT required format.
3. Keep the total response concise, helpful, and encouraging.
"""

    try:
        res = llm.invoke(f"{system_prompt}\n\n{user_prompt}")
        response_text = res.content if hasattr(res, 'content') else str(res)
    except Exception as e:
        print(f"LLM Chat Error: {e}")
        response_text = format_recommendation_fallback(top_schemes, user_profile)

    return {
        "retrieved_schemes": top_schemes,
        "response": response_text
    }
