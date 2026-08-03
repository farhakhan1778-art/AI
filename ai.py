"""Yojana Sahayak: Gemini RAG chatbot for Indian government schemes.

Pipeline:
  python yojana_sahayak_final.py test-api
  python yojana_sahayak_final.py scrape --limit 10
  python yojana_sahayak_final.py inspect
  python yojana_sahayak_final.py enrich
  python yojana_sahayak_final.py audit
  python yojana_sahayak_final.py index --reset
  python -m streamlit run yojana_sahayak_final.py

The scraper uses requests/BeautifulSoup for sitemap discovery and Playwright for
rendering JavaScript scheme pages. It does not call undocumented myScheme APIs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import time
import traceback
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import chromadb
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import find_dotenv, load_dotenv
from google import genai
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(usecwd=True), override=True)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
log = logging.getLogger("yojana_sahayak")

BASE_URL = os.getenv("MYSCHEME_BASE_URL", "https://www.myscheme.gov.in")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
DATA_FILE = DATA_DIR / "schemes.json"
CLEAN_FILE = DATA_DIR / "schemes_clean.txt"
AUDIT_FILE = DATA_DIR / "audit_report.json"
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma_db"))
COLLECTION = os.getenv("CHROMA_COLLECTION", "yojana_schemes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.6-flash")
METADATA_MODEL = os.getenv("METADATA_MODEL", "gemini-3.5-flash-lite")
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY_SECONDS", "1.0"))
PAGE_TIMEOUT_MS = int(os.getenv("PAGE_TIMEOUT_MS", "45000"))

FIELDS = ["state", "age", "gender", "occupation", "income", "category", "interest"]
QUESTIONS = {
    "state": "Which State or Union Territory do you live in?",
    "age": "How old are you?",
    "gender": "What is your gender?",
    "occupation": "What is your occupation?",
    "income": "What is your annual household income? You can enter 200000 or 2 LPA.",
    "category": "What is your social category, such as General, OBC, SC, ST, or EWS?",
    "interest": "What assistance are you looking for, such as farming, education, health, housing, or employment?",
}
SYSTEM_PROMPT = """You are Yojana Sahayak, a warm, concise assistant for Indian government schemes.
Use only RETRIEVED SCHEME EVIDENCE. Recommend at most 3 to 5 schemes. For every scheme include:
name, why it may match, eligibility conditions still requiring verification, benefits, how to apply,
and the verified myScheme link. Never invent information or claim guaranteed eligibility.
Exclude any record that contradicts the citizen profile."""


class SchemeMetadata(BaseModel):
    state: str = "All India"
    occupations: list[str] = Field(default_factory=list)
    gender: str = "All"
    minimum_age: int | None = None
    maximum_age: int | None = None
    maximum_income: float | None = None
    categories: list[str] = Field(default_factory=list)
    benefit_type: str = "Other"
    summary: str = ""


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", BeautifulSoup(str(value or ""), "html.parser").get_text(" ")).strip()


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def google_key() -> str:
    key = (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY") or "").strip()
    if not key or key.startswith("replace_"):
        raise RuntimeError("Set a real GOOGLE_API_KEY in .env.")
    return key


def available_generate_models() -> list[str]:
    client = genai.Client(api_key=google_key())
    names: list[str] = []
    try:
        for model in client.models.list():
            actions = set(getattr(model, "supported_actions", None) or [])
            name = str(getattr(model, "name", "")).replace("models/", "")
            if name and (not actions or "generateContent" in actions):
                names.append(name)
    finally:
        client.close()
    return names


def resolve_model(preferred: str, purpose: str) -> str:
    names = available_generate_models()
    if preferred in names:
        return preferred
    if purpose == "metadata":
        preferences = ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.6-flash", "gemini-3.5-flash"]
    else:
        preferences = ["gemini-3.6-flash", "gemini-3.5-flash", "gemini-3-flash", "gemini-3.5-flash-lite"]
    for candidate in preferences:
        if candidate in names:
            log.warning("Model %s unavailable; using %s", preferred, candidate)
            return candidate
    flash = [name for name in names if "flash" in name and "image" not in name and "live" not in name]
    if flash:
        log.warning("Model %s unavailable; using %s", preferred, flash[0])
        return flash[0]
    raise RuntimeError("No generateContent Gemini model is available to this API key.")


def make_llm(model: str, max_tokens: int) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model,
        api_key=google_key(),
        temperature=0,
        max_output_tokens=max_tokens,
        timeout=90,
        max_retries=2,
        vertexai=False,
    )


def response_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if text:
        return str(text)
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")) for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return str(content)


def test_api() -> None:
    metadata_model = resolve_model(METADATA_MODEL, "metadata")
    chat_model = resolve_model(CHAT_MODEL, "chat")
    reply = make_llm(metadata_model, 20).invoke("Reply exactly: API working")
    print(response_text(reply))
    print("Metadata model:", metadata_model)
    print("Chat model:", chat_model)


def load_rows() -> list[dict[str, Any]]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Missing {DATA_FILE}. Run scrape first.")
    raw = DATA_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        raise RuntimeError("data/schemes.json is blank.")
    try:
        rows = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid schemes.json at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("data/schemes.json contains zero schemes.")
    return [row for row in rows if isinstance(row, dict) and clean(row.get("scheme_name"))]


def save_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to overwrite schemes.json with zero records.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp = DATA_FILE.with_suffix(".tmp")
    temp.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(DATA_FILE)


def scheme_text(row: dict[str, Any]) -> str:
    keys = ["scheme_name", "description", "eligibility", "benefits", "required_documents",
            "ministry", "state", "application_process", "official_website"]
    body = "\n".join(f"{key.replace('_', ' ').title()}: {row.get(key, '')}" for key in keys)
    return body + "\nMetadata: " + json.dumps(row.get("metadata", {}), ensure_ascii=False)


def _extract_locations(text: str) -> list[str]:
    """Extract URLs from XML or browser-rendered sitemap text."""
    soup = BeautifulSoup(text, "xml")
    locations = [node.get_text(strip=True) for node in soup.find_all("loc")]
    if locations:
        return locations
    # Some WAF/CDN responses expose XML as plain text in a browser.
    return re.findall(r"https://www\.myscheme\.gov\.in/[^\s<>'\"]+", text)


def discover_scheme_urls(limit: int) -> list[str]:
    """Discover scheme pages with requests, then browser fallback.

    The myScheme sitemap may reject non-browser HTTP clients. A Playwright
    fallback reads the same official sitemap in Chromium. No undocumented API
    is used. If both methods fail, a clear diagnostic file is written.
    """
    root = urljoin(BASE_URL, "/sitemap.xml")
    pending = [root]
    visited: set[str] = set()
    scheme_urls: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    headers = {
        "User-Agent": os.getenv(
            "SCRAPER_USER_AGENT",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36",
        ),
        "Accept": "application/xml,text/xml,text/html;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-IN,en;q=0.9",
    }

    # Attempt 1: lightweight requests discovery.
    session = requests.Session()
    session.headers.update(headers)
    while pending and len(visited) < 150 and len(set(scheme_urls)) < limit:
        sitemap = pending.pop(0)
        if sitemap in visited:
            continue
        visited.add(sitemap)
        try:
            response = session.get(sitemap, timeout=30, allow_redirects=True)
            diagnostics.append({"method": "requests", "url": sitemap, "status": response.status_code,
                                "content_type": response.headers.get("content-type"), "bytes": len(response.content)})
            if response.status_code != 200:
                continue
            locations = _extract_locations(response.text)
            pending.extend(url for url in locations if re.search(r"\.xml(?:\.gz)?$", url, re.I) and url not in visited)
            scheme_urls.extend(url.split("?")[0].rstrip("/") for url in locations
                               if re.search(r"/schemes/[^/?#]+/?(?:\?.*)?$", url, re.I))
        except Exception as exc:
            diagnostics.append({"method": "requests", "url": sitemap, "error": str(exc)})

    # Attempt 2: Chromium fallback for WAF-protected or JS-served sitemap.
    if len(set(scheme_urls)) < limit:
        pending = [root]
        browser_visited: set[str] = set()
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(locale="en-IN", user_agent=headers["User-Agent"])
            page = context.new_page()
            while pending and len(browser_visited) < 150 and len(set(scheme_urls)) < limit:
                sitemap = pending.pop(0)
                if sitemap in browser_visited:
                    continue
                browser_visited.add(sitemap)
                try:
                    response = page.goto(sitemap, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
                    status = response.status if response else 0
                    page.wait_for_timeout(500)
                    # XML may be available as page content or as visible text.
                    content = page.content()
                    try:
                        content += "\n" + page.locator("body").inner_text(timeout=5000)
                    except Exception:
                        pass
                    diagnostics.append({"method": "playwright", "url": sitemap, "status": status,
                                        "final_url": page.url, "bytes": len(content)})
                    if status != 200:
                        continue
                    locations = _extract_locations(content)
                    pending.extend(url for url in locations if re.search(r"\.xml(?:\.gz)?$", url, re.I)
                                   and url not in browser_visited)
                    scheme_urls.extend(url.split("?")[0].rstrip("/") for url in locations
                                       if re.search(r"/schemes/[^/?#]+/?(?:\?.*)?$", url, re.I))
                except Exception as exc:
                    diagnostics.append({"method": "playwright", "url": sitemap, "error": str(exc)})
            context.close()
            browser.close()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "sitemap_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    unique = list(dict.fromkeys(scheme_urls))
    if not unique:
        statuses = [str(item.get("status")) for item in diagnostics if "status" in item]
        raise RuntimeError(
            "No scheme URLs could be read from the official sitemap. "
            f"Observed HTTP statuses: {', '.join(statuses) or 'none'}. "
            "See data/sitemap_diagnostics.json. If statuses are 403, the portal is blocking automated "
            "sitemap access from your network; do not fabricate scheme URLs."
        )
    return unique[:limit]


SECTION_ALIASES = {
    "description": ["details", "description", "about the scheme", "brief description"],
    "benefits": ["benefits"],
    "eligibility": ["eligibility", "eligibility criteria"],
    "application_process": ["application process", "how to apply"],
    "required_documents": ["documents required", "required documents", "documents"],
}
ALL_HEADINGS = {alias for aliases in SECTION_ALIASES.values() for alias in aliases} | {
    "frequently asked questions", "faqs", "sources and references", "feedback", "quick links"
}


def extract_text_section(lines: list[str], aliases: list[str]) -> str:
    start = None
    for index, line in enumerate(lines):
        if norm(line) in {norm(alias) for alias in aliases}:
            start = index + 1
            break
    if start is None:
        return ""
    output: list[str] = []
    for line in lines[start:]:
        if norm(line) in {norm(heading) for heading in ALL_HEADINGS}:
            break
        output.append(line)
    return clean(" ".join(output))


def parse_rendered_page(page: Any, url: str) -> dict[str, Any] | None:
    try:
        page.wait_for_selector("body", timeout=PAGE_TIMEOUT_MS)
        page.wait_for_timeout(2500)
        body = page.locator("body").inner_text(timeout=PAGE_TIMEOUT_MS)
    except PlaywrightTimeoutError:
        return None
    lines = [clean(line) for line in body.splitlines() if clean(line)]
    h1 = page.locator("h1").first
    name = clean(h1.inner_text()) if h1.count() else ""
    if not name:
        title = clean(page.title())
        name = re.sub(r"\s*[-|]\s*myScheme.*$", "", title, flags=re.I).strip()
    row: dict[str, Any] = {
        "scheme_name": name,
        "description": extract_text_section(lines, SECTION_ALIASES["description"]),
        "eligibility": extract_text_section(lines, SECTION_ALIASES["eligibility"]),
        "benefits": extract_text_section(lines, SECTION_ALIASES["benefits"]),
        "required_documents": extract_text_section(lines, SECTION_ALIASES["required_documents"]),
        "application_process": extract_text_section(lines, SECTION_ALIASES["application_process"]),
        "ministry": "",
        "state": "",
        "official_website": url,
        "source_url": url,
        "id": hashlib.sha256(url.encode()).hexdigest()[:24],
        "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Preserve visible text as fallback evidence, but never invent fields.
    if not row["description"] and len(body) > 500:
        row["description"] = clean(body[:4000])
    useful = sum(len(row[field]) for field in ("description", "eligibility", "benefits"))
    if not row["scheme_name"] or useful < 150:
        return None
    return row


def scrape(limit: int) -> None:
    urls = discover_scheme_urls(limit)
    if not urls:
        raise RuntimeError("The sitemap returned no scheme URLs.")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(locale="en-IN", user_agent=os.getenv("SCRAPER_USER_AGENT", "YojanaSahayak/1.0"))
        page = context.new_page()
        for number, url in enumerate(urls, 1):
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=PAGE_TIMEOUT_MS)
                status = response.status if response else 0
                final_url = page.url.split("?")[0]
                if status != 200 or "/schemes/" not in final_url:
                    failures.append({"url": url, "status": status, "reason": "invalid response or redirect"})
                    continue
                row = parse_rendered_page(page, final_url)
                if row:
                    rows.append(row)
                    log.info("Scraped %s/%s: %s", number, len(urls), row["scheme_name"])
                else:
                    failures.append({"url": url, "status": status, "reason": "insufficient rendered content"})
                time.sleep(REQUEST_DELAY)
            except Exception as exc:
                failures.append({"url": url, "status": 0, "reason": str(exc)})
                log.warning("Failed %s: %s", url, exc)
        context.close()
        browser.close()
    (DATA_DIR / "scrape_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    if not rows:
        raise RuntimeError("Rendered scraping extracted zero records. Existing schemes.json was not overwritten. See data/scrape_failures.json.")
    save_rows(rows)
    CLEAN_FILE.write_text("\n\n--- SCHEME ---\n\n".join(scheme_text(row) for row in rows), encoding="utf-8")
    print(f"Saved {len(rows)} verified pages; {len(failures)} failures. Inspect data/schemes.json before enrichment.")


def inspect_data() -> None:
    rows = load_rows()
    print("Records:", len(rows))
    for row in rows[:10]:
        print("\nNAME:", row.get("scheme_name"))
        print("LINK:", row.get("official_website"))
        print("ELIGIBILITY:", clean(row.get("eligibility"))[:250])
        print("BENEFITS:", clean(row.get("benefits"))[:250])


def enrich() -> None:
    test_api()
    rows = load_rows()
    model = resolve_model(METADATA_MODEL, "metadata")
    extractor = make_llm(model, 900).with_structured_output(SchemeMetadata)
    for index, row in enumerate(rows):
        if row.get("metadata"):
            continue
        try:
            result = extractor.invoke(
                "Extract only explicitly supported eligibility metadata. Use null or All when unspecified.\n\n" + scheme_text(row)
            )
            row["metadata"] = SchemeMetadata.model_validate(result).model_dump() if isinstance(result, dict) else result.model_dump()
            save_rows(rows)
            log.info("Enriched %s/%s", index + 1, len(rows))
        except Exception as exc:
            log.error("Enrichment failed for %s: %s", row.get("scheme_name"), exc)
    CLEAN_FILE.write_text("\n\n--- SCHEME ---\n\n".join(scheme_text(row) for row in rows), encoding="utf-8")


def dedupe_and_validate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, str]] = []
    for row in rows:
        name = clean(row.get("scheme_name"))
        url = clean(row.get("official_website"))
        parsed = urlparse(url)
        useful = sum(len(clean(row.get(field))) for field in ("description", "eligibility", "benefits"))
        if not name or parsed.netloc != "www.myscheme.gov.in" or not re.search(r"/schemes?/[^/]+/?$", parsed.path) or useful < 150:
            rejected.append({"name": name, "url": url, "reason": "invalid name, myScheme link, or insufficient source text"})
            continue
        key = parsed.path.rstrip("/").lower()
        if key not in unique or len(scheme_text(row)) > len(scheme_text(unique[key])):
            unique[key] = row
    report = {
        "source_records": len(rows),
        "valid_unique_records": len(unique),
        "duplicates_or_invalid_removed": len(rows) - len(unique),
        "rejected": rejected,
    }
    return list(unique.values()), report


def audit() -> None:
    _, report = dedupe_and_validate(load_rows())
    AUDIT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rejected"}, indent=2))
    print("Detailed report:", AUDIT_FILE)


def embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def scalar_metadata(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("metadata") or {}
    return {
        "scheme_name": clean(row.get("scheme_name")),
        "state": clean(meta.get("state") or row.get("state") or "All India"),
        "gender": clean(meta.get("gender") or "All"),
        "minimum_age": int(meta["minimum_age"]) if meta.get("minimum_age") is not None else -1,
        "maximum_age": int(meta["maximum_age"]) if meta.get("maximum_age") is not None else -1,
        "maximum_income": float(meta["maximum_income"]) if meta.get("maximum_income") is not None else -1.0,
        "occupation_csv": "|".join(meta.get("occupations") or []) or "All",
        "category_csv": "|".join(meta.get("categories") or []) or "All",
        "benefit_type": clean(meta.get("benefit_type") or "Other"),
        "official_website": clean(row.get("official_website")),
    }


def build_index(reset: bool) -> None:
    rows, report = dedupe_and_validate(load_rows())
    AUDIT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    enriched = [row for row in rows if row.get("metadata")]
    if not enriched:
        raise RuntimeError("No enriched valid records. Run enrich first.")
    if reset and CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    store = Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR), embedding_function=embeddings())
    documents = [Document(id=row["id"], page_content=scheme_text(row), metadata=scalar_metadata(row)) for row in enriched]
    store.add_documents(documents, ids=[document.id for document in documents])
    print(f"Indexed {len(documents)} unique, enriched, source-validated schemes.")


STATE_ALIASES = {"delhi": "Delhi", "new delhi": "Delhi", "nct of delhi": "Delhi", "orissa": "Odisha", "uttaranchal": "Uttarakhand"}
GENDER_ALIASES = {"m": "Male", "man": "Male", "male": "Male", "f": "Female", "woman": "Female", "female": "Female"}
CATEGORY_ALIASES = {"general": "General", "gen": "General", "obc": "OBC", "sc": "SC", "st": "ST", "ews": "EWS"}


def canonical(field: str, value: str) -> Any:
    value = " ".join(value.strip().split())
    key = value.lower()
    if field == "state": return STATE_ALIASES.get(key, value.title())
    if field == "gender": return GENDER_ALIASES.get(key, value.title())
    if field == "category": return CATEGORY_ALIASES.get(key, value.upper())
    if field == "occupation": return value.title()
    if field == "age":
        match = re.search(r"\d{1,3}", value)
        return int(match.group()) if match else value
    if field == "income":
        match = re.search(r"([\d.]+)", value.replace(",", ""))
        if not match: return value
        amount = float(match.group(1))
        lowered = value.lower()
        if "lpa" in lowered or "lakh" in lowered or "lac" in lowered: amount *= 100000
        elif "crore" in lowered or re.search(r"\bcr\b", lowered): amount *= 10000000
        return int(amount)
    return value


def hard_mismatch(document: Document, profile: dict[str, Any]) -> bool:
    meta = document.metadata
    user_state, scheme_state = norm(profile.get("state")), norm(meta.get("state"))
    if user_state and scheme_state not in {"", "all", "all india", "central", "pan india"} and user_state != scheme_state:
        return True
    user_gender, scheme_gender = norm(profile.get("gender")), norm(meta.get("gender"))
    if user_gender == "male" and scheme_gender in {"female", "woman", "women"}: return True
    if user_gender == "female" and scheme_gender in {"male", "man", "men"}: return True
    age = profile.get("age")
    if isinstance(age, int):
        if int(meta.get("minimum_age", -1)) >= 0 and age < int(meta["minimum_age"]): return True
        if int(meta.get("maximum_age", -1)) >= 0 and age > int(meta["maximum_age"]): return True
    income = profile.get("income")
    if isinstance(income, (int, float)) and float(meta.get("maximum_income", -1)) >= 0 and income > float(meta["maximum_income"]): return True
    category, categories = norm(profile.get("category")), norm(meta.get("category_csv"))
    if category and categories not in {"", "all", "any", "general"} and category not in categories: return True
    occupation, occupations = norm(profile.get("occupation")), norm(meta.get("occupation_csv"))
    if occupation and occupations not in {"", "all", "any", "general", "all occupations"} and occupation not in occupations: return True
    return False


def profile_score(document: Document, profile: dict[str, Any]) -> float:
    meta, content, score = document.metadata, norm(document.page_content), 0.0
    if norm(profile.get("state")) == norm(meta.get("state")): score += 8
    elif norm(meta.get("state")) in {"all", "all india", "central", "pan india"}: score += 4
    if norm(profile.get("occupation")) in norm(meta.get("occupation_csv")): score += 10
    if norm(profile.get("category")) in norm(meta.get("category_csv")): score += 7
    interest = norm(profile.get("interest"))
    if interest and interest in norm(meta.get("benefit_type")): score += 12
    if interest and interest in content: score += 6
    if norm(profile.get("occupation")) == "farmer" and any(term in content for term in ("farmer", "agriculture", "crop", "irrigation", "livestock", "kisan")): score += 12
    return score


def retrieve(profile: dict[str, Any], k: int = 10) -> list[Document]:
    store = Chroma(collection_name=COLLECTION, persist_directory=str(CHROMA_DIR), embedding_function=embeddings())
    count = store._collection.count()
    if not count: raise RuntimeError("Chroma is empty. Run index --reset.")
    query = "Indian government welfare schemes for " + "; ".join(f"{key}={value}" for key, value in profile.items() if value)
    pairs = store.similarity_search_with_relevance_scores(query, k=min(max(100, k * 10), count))
    seen: set[str] = set()
    ranked: list[tuple[Document, float]] = []
    for document, semantic in pairs:
        identity = document.metadata.get("official_website", "")
        if not identity or identity in seen or hard_mismatch(document, profile): continue
        seen.add(identity)
        ranked.append((document, profile_score(document, profile) + float(semantic)))
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [document for document, _ in ranked[:k]]


def recommend(profile: dict[str, Any], documents: list[Document]) -> str:
    model = resolve_model(CHAT_MODEL, "chat")
    evidence = "\n\n===== RETRIEVED SCHEME =====\n\n".join(document.page_content for document in documents)
    message = make_llm(model, 2200).invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"USER PROFILE:\n{json.dumps(profile, ensure_ascii=False)}\n\nRETRIEVED SCHEME EVIDENCE:\n{evidence}"),
    ])
    result = response_text(message).strip()
    if not result: raise RuntimeError("Gemini returned an empty response.")
    return result


def next_field(profile: dict[str, Any]) -> str | None:
    return next((field for field in FIELDS if not profile.get(field)), None)


def app() -> None:
    st.set_page_config(page_title="Yojana Sahayak", page_icon="🇮🇳")
    st.title("🇮🇳 Yojana Sahayak")
    st.caption("Recommendations are informational. Verify final eligibility on the linked myScheme page.")
    if "profile" not in st.session_state:
        st.session_state.profile = {field: None for field in FIELDS}
        st.session_state.messages = [{"role": "assistant", "content": QUESTIONS["state"]}]
        st.session_state.awaiting = "state"
        st.session_state.last_error = None
    with st.sidebar:
        st.subheader("Profile")
        st.json({key: value for key, value in st.session_state.profile.items() if value})
        try: count = chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection(COLLECTION).count()
        except Exception: count = 0
        st.metric("Indexed schemes", count)
        if st.session_state.last_error:
            with st.expander("Last technical error"): st.code(st.session_state.last_error)
        if st.button("Start over"):
            for item in ("profile", "messages", "awaiting", "last_error"): st.session_state.pop(item, None)
            st.rerun()
    for message in st.session_state.messages:
        with st.chat_message(message["role"]): st.markdown(message["content"])
    user = st.chat_input("Type your answer")
    if not user: return
    st.session_state.messages.append({"role": "user", "content": user})
    current = st.session_state.awaiting
    if current: st.session_state.profile[current] = canonical(current, user)
    missing = next_field(st.session_state.profile)
    try:
        documents = retrieve(st.session_state.profile, 10)
        status = f"Thank you, noted. {len(documents)} eligible retrieval candidates are currently available."
        st.session_state.last_error = None
    except Exception as exc:
        documents = []
        status = f"Retrieval failed: `{type(exc).__name__}: {exc}`"
        st.session_state.last_error = traceback.format_exc()
    if missing:
        response = status + "\n\n" + QUESTIONS[missing]
    elif documents:
        try: response = status + "\n\n" + recommend(st.session_state.profile, documents)
        except Exception as exc:
            st.session_state.last_error = traceback.format_exc()
            response = status + f"\n\nGemini ranking failed: `{type(exc).__name__}: {exc}`"
    else:
        response = status + "\n\nNo scheme passed the current filters."
    st.session_state.awaiting = missing
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    scrape_command = sub.add_parser("scrape")
    scrape_command.add_argument("--limit", type=int, default=250)
    sub.add_parser("inspect")
    sub.add_parser("test-api")
    sub.add_parser("enrich")
    sub.add_parser("audit")
    index_command = sub.add_parser("index")
    index_command.add_argument("--reset", action="store_true")
    args = parser.parse_args()
    if args.command == "scrape": scrape(args.limit)
    elif args.command == "inspect": inspect_data()
    elif args.command == "test-api": test_api()
    elif args.command == "enrich": enrich()
    elif args.command == "audit": audit()
    elif args.command == "index": build_index(args.reset)
    else: app()


if __name__ == "__main__":
    main()