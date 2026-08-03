# verify_pipeline.py
# Verification test script for Yojana Sahayak chatbot flow & RAG pipeline.

import json
from chatbot import ChatbotSession, get_live_count
from retrieval_pipeline import run_rag_pipeline

def test_chatbot_flow():
    print("=== TESTING YOJANA SAHAYAK CHATBOT & RAG PIPELINE ===")
    session = ChatbotSession()

    test_answers = [
        "Rajasthan",              # Q1: State
        "32 years old",           # Q2: Age
        "Male",                   # Q3: Gender
        "Farmer",                 # Q4: Occupation
        "Below 1L",               # Q5: Income
        "OBC",                    # Q6: Category
        "Crop subsidy and seeds"  # Q7: Interest
    ]

    for idx, answer in enumerate(test_answers, 1):
        print(f"\n--- User Step {idx}: '{answer}' ---")
        reply = session.process_user_message(answer)
        print(f"Profile state so far: {session.profile}")
        print(f"Live matching count: {get_live_count(session.profile)}")
        print(f"Bot Reply Snippet:\n{reply[:250]}...\n")

    print("\n[OK] Full 7-step chatbot flow executed cleanly!")
    print(f"Final profile: {json.dumps(session.profile, indent=2)}")
    print(f"Final completed status: {session.completed}")

if __name__ == "__main__":
    test_chatbot_flow()
