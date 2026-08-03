# create_demo_video.py
# Generates a 1080p 30fps MP4 demonstration video for Yojana Sahayak.

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Target output MP4 video path in artifacts directory
ARTIFACT_DIR = Path(r"C:\Users\MSI\.gemini\antigravity\brain\ccad62c2-73e0-4418-a9f3-c67d5b8c2df2")
OUTPUT_VIDEO_PATH = ARTIFACT_DIR / "yojana_sahayak_demo.mp4"

WIDTH, HEIGHT = 1920, 1080
FPS = 30

def get_fonts():
    try:
        title_font = ImageFont.truetype("arial.ttf", 60)
        subtitle_font = ImageFont.truetype("arial.ttf", 32)
        body_font = ImageFont.truetype("arial.ttf", 26)
        small_font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    return title_font, subtitle_font, body_font, small_font

def draw_header(draw, title_font, subtitle_font, title_text, sub_text):
    # Dark background gradient
    draw.rectangle([0, 0, WIDTH, 120], fill=(13, 17, 23))
    draw.rectangle([0, 118, WIDTH, 120], fill=(16, 185, 129))
    draw.text((60, 25), title_text, fill=(16, 185, 129), font=title_font)
    draw.text((60, 80), sub_text, fill=(148, 163, 184), font=subtitle_font)

def create_scene_1(title_font, subtitle_font, body_font):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(7, 11, 20))
    draw = ImageDraw.Draw(img)

    # Title Card
    draw.rectangle([100, 200, 1820, 880], fill=(13, 17, 23), outline=(16, 185, 129), width=2)
    draw.text((160, 260), "YOJANA SAHAYAK", fill=(16, 185, 129), font=title_font)
    draw.text((160, 340), "Standard RAG AI Chatbot for Indian Government Schemes", fill=(241, 245, 249), font=subtitle_font)
    
    y = 440
    tech_stack = [
        "• Web Scraping & Dataset: requests, BeautifulSoup (250+ schemes)",
        "• Metadata Extraction: LLM Structured JSON Enrichment",
        "• Vector Database: ChromaDB + sentence-transformers (all-MiniLM-L6-v2)",
        "• Retrieval Pipeline: LangChain RAG + Metadata Filtering + LLM Ranking",
        "• Frontend Interface: Streamlit Dark Glassmorphism UI",
        "• Clean Architecture: Completely free of Neo4j, Graph RAG, or FAISS"
    ]
    for line in tech_stack:
        draw.text((160, y), line, fill=(226, 232, 240), font=body_font)
        y += 50
    return np.array(img)

def create_scene_2(title_font, subtitle_font, body_font, small_font):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(7, 11, 20))
    draw = ImageDraw.Draw(img)
    draw_header(draw, title_font, subtitle_font, "SYSTEM ARCHITECTURE", "End-to-End Standard RAG Pipeline Flow")

    # Workflow blocks
    blocks = [
        ("1. Data Scraping", "250+ Schemes\nmyscheme.gov.in\ndata/schemes.json", 80, 300, (16, 185, 129)),
        ("2. Metadata Extraction", "Structured JSON\nState, Occupation,\nIncome, Category", 440, 300, (59, 130, 246)),
        ("3. ChromaDB Indexing", "Dense Vectors (384-d)\nall-MiniLM-L6-v2\nMetadata Filters", 800, 300, (168, 85, 247)),
        ("4. 7-Step Chatbot", "Dynamic Questionnaire\nLive Match Counter\nUser Profile State", 1160, 300, (245, 158, 11)),
        ("5. LLM Recommendations", "LangChain + Sonnet/Gemini\nTop 3-5 Ranked Cards\nOfficial Portal Links", 1520, 300, (236, 72, 153))
    ]

    for title, desc, x, y, col in blocks:
        draw.rectangle([x, y, x + 320, y + 420], fill=(13, 17, 23), outline=col, width=3)
        draw.rectangle([x, y, x + 320, y + 60], fill=col)
        draw.text((x + 15, y + 15), title, fill=(255, 255, 255), font=subtitle_font)
        
        dy = y + 90
        for line in desc.split("\n"):
            draw.text((x + 20, dy), line, fill=(226, 232, 240), font=body_font)
            dy += 40

    # Key features box
    draw.rectangle([80, 800, 1840, 1000], fill=(13, 17, 23), outline=(16, 185, 129), width=1)
    draw.text((110, 820), "Core Technical Capabilities:", fill=(16, 185, 129), font=subtitle_font)
    draw.text((110, 870), "✓ Strict metadata pre-filtering  ✓ Live matching scheme counter  ✓ Turn-by-turn question flow  ✓ Full eligibility explanation", fill=(226, 232, 240), font=body_font)

    return np.array(img)

def create_scene_3(title_font, subtitle_font, body_font, small_font, progress_pct=71, match_count=15):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(7, 11, 20))
    draw = ImageDraw.Draw(img)
    draw_header(draw, title_font, subtitle_font, "YOJANA SAHAYAK — DEMO INTERFACE", "Split-Panel Streamlit Dashboard & Chatbot")

    # Left Panel
    draw.rectangle([40, 150, 520, 1020], fill=(13, 17, 23), outline=(30, 41, 59), width=2)
    draw.text((70, 180), "Yojana Sahayak", fill=(16, 185, 129), font=title_font)
    draw.text((70, 240), "Standard RAG · AI Discovery", fill=(100, 116, 139), font=small_font)
    
    # Progress
    draw.text((70, 290), f"PROFILE PROGRESS — {progress_pct}%", fill=(148, 163, 184), font=small_font)
    draw.rectangle([70, 320, 490, 330], fill=(30, 41, 59))
    fill_w = int(70 + (420 * (progress_pct / 100)))
    draw.rectangle([70, 320, fill_w, 330], fill=(16, 185, 129))

    # Profile Grid
    draw.text((70, 360), "CITIZEN PROFILE", fill=(148, 163, 184), font=small_font)
    profile_data = [
        ("State/UT", "Rajasthan"),
        ("Age", "32"),
        ("Gender", "Male"),
        ("Occupation", "Farmer"),
        ("Income", "Below 1L"),
        ("Category", "OBC"),
        ("Interest", "Crop & Seeds")
    ]
    py = 390
    for k, v in profile_data:
        draw.rectangle([70, py, 490, py + 36], fill=(15, 23, 42), outline=(30, 41, 59))
        draw.text((85, py + 6), k, fill=(148, 163, 184), font=small_font)
        draw.text((340, py + 6), v, fill=(16, 185, 129), font=small_font)
        py += 44

    # Match Counter Badge
    draw.rectangle([70, 720, 490, 840], fill=(6, 78, 59), outline=(16, 185, 129), width=2)
    draw.text((220, 735), f"{match_count}", fill=(52, 211, 153), font=title_font)
    draw.text((120, 800), "MATCHING SCHEMES FOUND", fill=(148, 163, 184), font=small_font)

    # Right Panel - Chat Timeline
    draw.rectangle([550, 150, 1880, 1020], fill=(13, 17, 23), outline=(30, 41, 59), width=2)
    draw.rectangle([550, 150, 1880, 210], fill=(15, 23, 42))
    draw.text((580, 170), "💬 Conversation with Yojana Sahayak", fill=(241, 245, 249), font=subtitle_font)

    # Chat Messages
    msgs = [
        ("bot", "Namaste! I am Yojana Sahayak. Which State or Union Territory do you live in?"),
        ("user", "Rajasthan"),
        ("bot", "Found 40 matching schemes so far. How old are you?"),
        ("user", "32 years old"),
        ("bot", "Found 27 matching schemes so far. What is your primary occupation?"),
        ("user", "Farmer"),
        ("bot", f"Found {match_count} matching schemes so far.\n\nNamaste! Here are your top matching schemes:")
    ]
    my = 230
    for role, text in msgs:
        if role == "user":
            draw.rectangle([1300, my, 1850, my + 50], fill=(30, 58, 138), outline=(59, 130, 246))
            draw.text((1320, my + 10), text, fill=(239, 246, 255), font=body_font)
            my += 65
        else:
            draw.rectangle([580, my, 1350, my + 70], fill=(15, 23, 42), outline=(30, 41, 59))
            for i, line in enumerate(text.split("\n")):
                draw.text((600, my + 10 + (i * 26)), line, fill=(226, 232, 240), font=body_font)
            my += 85

    return np.array(img)

def create_scene_4(title_font, subtitle_font, body_font, small_font):
    img = Image.new("RGB", (WIDTH, HEIGHT), color=(7, 11, 20))
    draw = ImageDraw.Draw(img)
    draw_header(draw, title_font, subtitle_font, "TOP RECOMMENDED SCHEMES", "RAG Output: Ranked Cards with Eligibility & Application Steps")

    cards = [
        ("#1 MATCH", "PM-Kisan Samman Nidhi Yojana (Central)", "Farmer in Rajasthan", "₹6,000 / year direct bank transfer in 3 installments", "Apply online with Aadhaar at pmkisan.gov.in or nearest CSC center.", "https://pmkisan.gov.in", (16, 185, 129)),
        ("#2 MATCH", "Pradhan Mantri Crop Insurance Yojana (Central)", "Farmer with crop land", "Subsidized crop insurance cover against yield loss and natural risks", "Submit crop details on PMFBY portal or through local bank branch.", "https://pmfby.gov.in", (59, 130, 246)),
        ("#3 MATCH", "Mukhyamantri Kisan Pension Yojana (Rajasthan)", "Farmer age 32 in Rajasthan", "Monthly pension support of ₹1,000 for small & marginal farmers", "Submit form at Tehsil office or Rajasthan e-Mitra portal.", "https://raj.gov.in", (245, 158, 11))
    ]

    cy = 160
    for rank, name, qual, ben, app, link, col in cards:
        draw.rectangle([100, cy, 1820, cy + 250], fill=(13, 17, 23), outline=col, width=2)
        draw.rectangle([1650, cy + 15, 1800, cy + 50], fill=col)
        draw.text((1665, cy + 22), rank, fill=(255, 255, 255), font=small_font)
        
        draw.text((140, cy + 20), name, fill=(240, 253, 244), font=subtitle_font)
        draw.text((140, cy + 75), f"✅ Why you qualify: {qual}", fill=(226, 232, 240), font=body_font)
        draw.text((140, cy + 115), f"💰 Benefits: {ben}", fill=(226, 232, 240), font=body_font)
        draw.text((140, cy + 155), f"📋 How to apply: {app}", fill=(226, 232, 240), font=body_font)
        
        draw.rectangle([140, cy + 195, 320, cy + 230], fill=col)
        draw.text((155, cy + 202), "🔗 Official Link", fill=(255, 255, 255), font=small_font)
        draw.text((340, cy + 204), link, fill=(148, 163, 184), font=small_font)
        
        cy += 280

    return np.array(img)

def generate_video():
    print(f"Generating 1080p MP4 video at: {OUTPUT_VIDEO_PATH}")
    title_font, subtitle_font, body_font, small_font = get_fonts()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, FPS, (WIDTH, HEIGHT))

    # Scene durations (in seconds)
    # Scene 1: 4s, Scene 2: 5s, Scene 3: 6s, Scene 4: 6s, Scene 5 (Title end): 3s
    
    scene1_img = create_scene_1(title_font, subtitle_font, body_font)
    for _ in range(4 * FPS):
        out.write(cv2.cvtColor(scene1_img, cv2.COLOR_RGB2BGR))

    scene2_img = create_scene_2(title_font, subtitle_font, body_font, small_font)
    for _ in range(5 * FPS):
        out.write(cv2.cvtColor(scene2_img, cv2.COLOR_RGB2BGR))

    # Animated Scene 3 (Progressing profile)
    for t in range(6 * FPS):
        prog = min(100, int(20 + (t / (6 * FPS)) * 80))
        matches = max(14, int(250 - (t / (6 * FPS)) * 236))
        s3_img = create_scene_3(title_font, subtitle_font, body_font, small_font, progress_pct=prog, match_count=matches)
        out.write(cv2.cvtColor(s3_img, cv2.COLOR_RGB2BGR))

    scene4_img = create_scene_4(title_font, subtitle_font, body_font, small_font)
    for _ in range(6 * FPS):
        out.write(cv2.cvtColor(scene4_img, cv2.COLOR_RGB2BGR))

    # Closing scene: 3s
    closing_img = create_scene_1(title_font, subtitle_font, body_font)
    for _ in range(3 * FPS):
        out.write(cv2.cvtColor(closing_img, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"[OK] Video successfully created and saved to {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    generate_video()
