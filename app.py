import os
from dataclasses import dataclass
from datetime import datetime
import io
import json
import re
from typing import List, Dict, Any, Tuple
from urllib.parse import quote_plus

import feedparser
from pypdf import PdfReader
import streamlit as st
from ddgs import DDGS
from dotenv import load_dotenv


load_dotenv()


try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    AGNO_AVAILABLE = True
except Exception:
    AGNO_AVAILABLE = False


DEFAULT_NEWS_RSS = []

JOB_SITE_HINTS = [
    "site:boards.greenhouse.io",
    "site:jobs.lever.co",
    "site:jobs.smartrecruiters.com",
    "site:careers",
]


@dataclass
class Profile:
    role: str
    industry: str
    skills: List[str]
    location: str
    seniority: str
    keywords: List[str]
    exclusions: List[str]


def normalize_list(value: str) -> List[str]:
    if not value.strip():
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def build_google_news_rss(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote_plus(query)}"


def fetch_rss(url: str) -> List[Dict[str, Any]]:
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries:
        items.append(
            {
                "title": entry.get("title", ""),
                "url": entry.get("link", ""),
                "published": entry.get("published", ""),
                "source": feed.feed.get("title", "RSS"),
                "summary": entry.get("summary", ""),
            }
        )
    return items


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []
    for page in reader.pages:
        text = page.extract_text() or ""
        chunks.append(text)
    return "\n".join(chunks).strip()


def parse_json_block(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}


def ddg_search(query: str, max_results: int = 20) -> List[Dict[str, Any]]:
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "published": "",
                    "source": "DuckDuckGo",
                    "summary": r.get("body", ""),
                }
            )
    return results


def dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        key = (item.get("title", "").lower(), item.get("url", "").lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def score_item(item: Dict[str, Any], profile: Profile) -> Tuple[int, List[str]]:
    text = " ".join(
        [
            item.get("title", ""),
            item.get("summary", ""),
            profile.role,
            profile.industry,
        ]
    ).lower()
    signals = []
    score = 0

    for kw in profile.keywords + profile.skills:
        if kw.lower() in text:
            score += 2
            signals.append(kw)
    if profile.location and profile.location.lower() in text:
        score += 1
        signals.append(profile.location)
    if profile.industry and profile.industry.lower() in text:
        score += 1
        signals.append(profile.industry)

    for ex in profile.exclusions:
        if ex.lower() in text:
            score -= 3
            signals.append(f"-{ex}")

    return score, signals[:5]


def rank_items(items: List[Dict[str, Any]], profile: Profile) -> List[Dict[str, Any]]:
    ranked = []
    for item in items:
        score, signals = score_item(item, profile)
        item["score"] = score
        item["signals"] = signals
        ranked.append(item)
    ranked.sort(key=lambda x: x.get("score", 0), reverse=True)
    return ranked


def make_agent(model_id: str) -> Any:
    if not AGNO_AVAILABLE:
        return None
    return Agent(
        model=OpenAIChat(
            id=model_id,
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
        instructions=[
            "You are an industry news and job scout.",
            "Given a user profile and a single item, explain why it matches.",
            "Keep responses short: 1-2 sentences, no bullets.",
        ],
        markdown=False,
    )


def enrich_with_agent(agent: Any, profile: Profile, items: List[Dict[str, Any]]) -> None:
    if not agent:
        return
    for item in items:
        prompt = (
            "User profile:\n"
            f"- Role: {profile.role}\n"
            f"- Industry: {profile.industry}\n"
            f"- Skills: {', '.join(profile.skills)}\n"
            f"- Location: {profile.location}\n"
            f"- Seniority: {profile.seniority}\n"
            f"- Keywords: {', '.join(profile.keywords)}\n\n"
            f"Item:\nTitle: {item.get('title','')}\n"
            f"Summary: {item.get('summary','')}\n"
            f"URL: {item.get('url','')}\n\n"
            "Explain why this item matches the user."
        )
        try:
            response = agent.run(prompt)
            text = getattr(response, "content", None) or getattr(response, "output", None)
            item["why"] = (text or str(response)).strip()
        except Exception as exc:
            item["why"] = f"LLM unavailable: {exc}"


def analyze_resume(agent: Any, resume_text: str, inputs: Profile) -> Dict[str, Any]:
    if not agent or not resume_text.strip():
        return {}
    prompt = (
        "Extract structured insights from this resume and optional inputs.\n"
        "Return JSON only with keys: inferred_role, keywords, suggested_companies.\n"
        "Each suggested_companies item should be an object with name and reason.\n"
        "Keep keywords to 8-12 short phrases.\n\n"
        f"Inputs:\n"
        f"- Role: {inputs.role}\n"
        f"- Industry: {inputs.industry}\n"
        f"- Skills: {', '.join(inputs.skills)}\n"
        f"- Location: {inputs.location}\n"
        f"- Seniority: {inputs.seniority}\n"
        f"- Keywords: {', '.join(inputs.keywords)}\n"
        f"- Exclusions: {', '.join(inputs.exclusions)}\n\n"
        f"Resume:\n{resume_text}\n"
    )
    try:
        response = agent.run(prompt)
        text = getattr(response, "content", None) or getattr(response, "output", None)
        return parse_json_block(text or str(response))
    except Exception:
        return {}


st.set_page_config(page_title="Industry News + Jobs Agent", layout="wide")
st.title("Industry News + Jobs Agent")

with st.sidebar:
    st.header("Profile")
    role = st.text_input("Target role", "")
    industry = st.text_input("Industry", "")
    skills = st.text_input("Skills (comma)", "")
    location = st.text_input("Location", "")
    seniority = st.selectbox("Seniority", ["", "Junior", "Mid", "Senior", "Lead"])
    keywords = st.text_input("Keywords (comma)", "")
    exclusions = st.text_input("Exclude terms (comma)", "")
    st.divider()
    resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
    st.divider()
    use_agent = st.checkbox("Use Agno + OpenAI for match explanations", value=True)
    model_id = st.text_input("OpenAI model id", "gpt-4o-mini")


profile = Profile(
    role=role.strip(),
    industry=industry.strip(),
    skills=normalize_list(skills),
    location=location.strip(),
    seniority=seniority.strip(),
    keywords=normalize_list(keywords),
    exclusions=normalize_list(exclusions),
)


@st.cache_data(show_spinner=False)
def collect_news(profile: Profile, rss_urls: List[str], query: str) -> List[Dict[str, Any]]:
    items = []
    for url in rss_urls:
        items.extend(fetch_rss(url))
    if query.strip():
        items.extend(fetch_rss(build_google_news_rss(query.strip())))
    items = dedupe_items(items)
    return rank_items(items, profile)


@st.cache_data(show_spinner=False)
def build_job_queries(profile: Profile, companies: List[Dict[str, Any]]) -> List[str]:
    role_terms = profile.role or " ".join(profile.keywords)
    location = profile.location
    queries = []

    if companies:
        for company in companies:
            name = company.get("name", "").strip()
            if not name:
                continue
            query = f"{name} careers {role_terms} {location}".strip()
            queries.append(query)
    else:
        base_query = f"{role_terms} {location} {profile.industry}".strip()
        hints = " OR ".join(JOB_SITE_HINTS)
        queries.append(f'{base_query} ({hints})')

    return [q for q in queries if q.strip()]


def collect_jobs(profile: Profile, companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for query in build_job_queries(profile, companies):
        items.extend(ddg_search(query, max_results=15))
    items = dedupe_items(items)
    return rank_items(items, profile)


if st.button("Run search"):
    resume_text = ""
    if resume_file is not None:
        resume_text = extract_pdf_text(resume_file.getvalue())

    agent = make_agent(model_id) if use_agent else None
    resume_insights = analyze_resume(agent, resume_text, profile)
    inferred_role = resume_insights.get("inferred_role", "")
    resume_keywords = resume_insights.get("keywords", [])
    suggested_companies = resume_insights.get("suggested_companies", [])

    rss_urls = list(DEFAULT_NEWS_RSS)
    query_parts = [
        profile.industry or "",
        inferred_role or "",
        " ".join(profile.keywords),
        " ".join(resume_keywords),
    ]
    news_query = " ".join([p for p in query_parts if p]).strip()
    if not news_query:
        st.warning("Add an industry, keywords, or upload a resume to fetch news.")
        st.stop()

    effective_profile = Profile(
        role=profile.role or inferred_role,
        industry=profile.industry,
        skills=profile.skills,
        location=profile.location,
        seniority=profile.seniority,
        keywords=profile.keywords or resume_keywords,
        exclusions=profile.exclusions,
    )
    if not effective_profile.role and not effective_profile.keywords:
        st.warning("Add a target role/keywords or upload a resume to fetch jobs.")
        st.stop()

    with st.spinner("Fetching news..."):
        news_items = collect_news(profile, rss_urls, news_query)

    with st.spinner("Fetching jobs..."):
        job_items = collect_jobs(effective_profile, suggested_companies)

    if use_agent:
        if not agent:
            st.warning("Agno not available. Install requirements and restart.")
        else:
            with st.spinner("Explaining matches..."):
                enrich_with_agent(agent, profile, news_items[:10])
                enrich_with_agent(agent, profile, job_items[:10])

    col1, col2 = st.columns(2)
    with col1:
        if suggested_companies:
            st.subheader("Suggested Companies")
            for company in suggested_companies[:10]:
                name = company.get("name", "").strip()
                reason = company.get("reason", "").strip()
                if name:
                    st.markdown(f"**{name}**")
                    if reason:
                        st.write(reason)
                    st.divider()
        st.subheader("Industry News")
        for item in news_items[:15]:
            st.markdown(f"**{item.get('title','')}**")
            st.caption(f"{item.get('source','')} • {item.get('published','')}")
            if item.get("why"):
                st.write(item["why"])
            else:
                st.write(f"Signals: {', '.join(item.get('signals', []))}")
            st.link_button("Open", item.get("url", ""))
            st.divider()

    with col2:
        st.subheader("Job Listings")
        if not job_items:
            st.info("No job listings found. Try adding a role or uploading a resume.")
        for item in job_items[:15]:
            st.markdown(f"**{item.get('title','')}**")
            st.caption(f"{item.get('source','')} • {item.get('published','')}")
            if item.get("why"):
                st.write(item["why"])
            else:
                st.write(f"Signals: {', '.join(item.get('signals', []))}")
            st.link_button("Open", item.get("url", ""))
            st.divider()

else:
    st.info("Fill out the profile and click Run search.")
