# chatbot.py
# Turn-by-Turn Dynamic Questionnaire Manager & Conversation State Controller for Yojana Sahayak.

import json
from typing import Dict, Any, List, Optional
import config
from retrieval_pipeline import run_rag_pipeline

# 7-step sequence required by specification
QUESTIONS = [
    {
        "field": "state",
        "question": "Namaste! I am Yojana Sahayak. Which State or Union Territory do you live in?",
        "options": ["Rajasthan", "Uttar Pradesh", "Delhi", "Madhya Pradesh", "Maharashtra", "Bihar", "Central", "Other"]
    },
    {
        "field": "age",
        "question": "Dhanyawad! How old are you? (Please specify your age in years)",
        "options": []
    },
    {
        "field": "gender",
        "question": "Got it. What is your gender?",
        "options": ["Male", "Female", "Transgender"]
    },
    {
        "field": "occupation",
        "question": "What is your primary occupation?",
        "options": ["Farmer", "Student", "Salaried", "Business", "Unemployed"]
    },
    {
        "field": "income",
        "question": "What is your annual household income?",
        "options": ["Below 1L", "1-3L", "3-6L", "6-10L", "Above 10L"]
    },
    {
        "field": "category",
        "question": "What is your social category?",
        "options": ["General", "OBC", "SC", "ST", "Minority", "PWD"]
    },
    {
        "field": "interest",
        "question": "What kind of assistance or scheme are you looking for? (e.g., Agriculture, Education, Healthcare, Housing, Business, Financial, or type 'All')",
        "options": ["Agriculture", "Education", "Healthcare", "Housing", "Business", "Financial", "All"]
    }
]

def load_schemes_json() -> List[Dict[str, Any]]:
    """Loads schemes from data/schemes.json safely."""
    if not config.RAW_SCHEMES_PATH.exists():
        return []
    try:
        with open(config.RAW_SCHEMES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def get_live_count(profile: Dict[str, Any]) -> int:
    """Calculates the count of schemes matching the user's profile parameters so far."""
    schemes = load_schemes_json()
    if not schemes:
        return 0

    count = 0
    u_state = profile.get("state")
    u_age   = profile.get("age")
    u_gender= profile.get("gender")
    u_occ   = profile.get("occupation")
    u_inc   = profile.get("income")
    u_cat   = profile.get("category")

    for s in schemes:
        elig = s.get("eligibility", {})

        # State check
        if u_state and u_state.lower() not in ["all", "other", "central", "none"]:
            s_state = s.get("state", "central").lower()
            if s_state not in [u_state.lower(), "central"]:
                continue

        # Age check
        if u_age is not None and isinstance(u_age, (int, float)):
            if not (elig.get("age_min", 0) <= u_age <= elig.get("age_max", 100)):
                continue

        # Gender check
        if u_gender and u_gender.lower() in ["male", "female", "transgender"]:
            genders = [g.lower() for g in elig.get("genders", ["all"])]
            if u_gender.lower() not in genders and "all" not in genders:
                continue

        # Occupation check
        if u_occ and u_occ.lower() not in ["all", "other", "none"]:
            occs = [o.lower() for o in elig.get("occupations", ["all"])]
            if u_occ.lower() not in occs and "all" not in occs:
                continue

        # Category check
        if u_cat and u_cat.lower() not in ["all", "general", "other", "none"]:
            cats = [c.lower() for c in elig.get("categories", ["all"])]
            if u_cat.lower() not in cats and "all" not in cats and "general" not in cats:
                continue

        count += 1
    return count

def count_filled_fields(profile: Dict[str, Any]) -> int:
    """Returns number of non-null profile fields."""
    return sum(1 for v in profile.values() if v is not None and v != [] and v != "")

class ChatbotSession:
    """Manages profile accumulation, step sequence, and RAG execution."""
    def __init__(self):
        self.profile = {
            "state": None,
            "age": None,
            "gender": None,
            "occupation": None,
            "income": None,
            "category": None,
            "interest": None
        }
        self.step = 0
        self.history = []
        self.completed = False

    def process_user_message(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})
        
        # 1. Update current step field in profile
        if self.step < len(QUESTIONS):
            field_name = QUESTIONS[self.step]["field"]
            val = user_message.strip()

            if field_name == "age":
                digits = "".join(filter(str.isdigit, val))
                if digits:
                    self.profile["age"] = int(digits)
                else:
                    self.profile["age"] = 30 # default reasonable estimate
            elif field_name == "income":
                # Normalize income string format
                self.profile["income"] = val
            else:
                self.profile[field_name] = val

            self.step += 1

        filled_count = count_filled_fields(self.profile)
        live_count = get_live_count(self.profile)

        # 2. Build acknowledge and match count text
        ack_text = f"Found {live_count} matching schemes so far. "

        # 3. If filled_count >= 4 or step >= 4, execute RAG recommendations
        rec_text = ""
        if filled_count >= 4 or self.step >= 4:
            history_str = "\n".join([f"{m['role']}: {m['content']}" for m in self.history[-6:]])
            rag_res = run_rag_pipeline(
                user_query=user_message,
                user_profile=self.profile,
                chat_history_str=history_str
            )
            rec_text = rag_res.get("response", "")

        # 4. Formulate bot response text with next question if available
        if self.step < len(QUESTIONS):
            next_q = QUESTIONS[self.step]["question"]
            if rec_text:
                bot_response = f"{ack_text}\n\n{rec_text}\n\nTo refine your matches further: {next_q}"
            else:
                bot_response = f"{ack_text}{next_q}"
        else:
            self.completed = True
            if rec_text:
                bot_response = f"Bahut accha! Your profile is complete.\n\n{rec_text}"
            else:
                bot_response = f"Bahut accha! Your profile is complete. Found {live_count} schemes."

        self.history.append({"role": "assistant", "content": bot_response})
        return bot_response
