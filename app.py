import os
from dataclasses import dataclass
import io
import json
import re
from typing import List, Dict, Any
from urllib.parse import quote_plus

import feedparser
import httpx
from pypdf import PdfReader
import streamlit as st
from dotenv import load_dotenv


load_dotenv()


try:
    from agno.agent import Agent
    from agno.models.openai import OpenAIChat

    AGNO_AVAILABLE = True
except Exception:
    AGNO_AVAILABLE = False


DEFAULT_NEWS_RSS = []


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


SUPPORTED_ATS = {"lever", "greenhouse", "smartrecruiters"}


def fetch_lever_jobs(slug: str) -> List[Dict[str, Any]]:
    url = f"https://api.lever.co/v0/postings/{slug}?mode=json"
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return []


def fetch_greenhouse_jobs(slug: str) -> List[Dict[str, Any]]:
    url = f"https://boards-api.greenhouse.io/v1/boards/{slug}/jobs"
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("jobs", [])
    except Exception:
        return []


def fetch_smartrecruiters_jobs(slug: str) -> List[Dict[str, Any]]:
    url = f"https://api.smartrecruiters.com/v1/companies/{slug}/postings"
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("content", [])
    except Exception:
        return []


def normalize_role_terms(role: str, keywords: List[str]) -> List[str]:
    terms = []
    if role:
        terms.extend(role.lower().split())
    for kw in keywords:
        terms.extend(kw.lower().split())
    return [t for t in terms if t]


def title_matches(title: str, terms: List[str]) -> bool:
    title_lower = title.lower()
    return any(t in title_lower for t in terms)


def find_job_url_from_ats(company: Dict[str, Any], role: str, keywords: List[str]) -> str:
    ats = (company.get("ats") or "").lower().strip()
    slug = (company.get("ats_slug") or "").strip()
    if ats not in SUPPORTED_ATS or not slug:
        return ""
    terms = normalize_role_terms(role, keywords)
    jobs = []
    if ats == "lever":
        jobs = fetch_lever_jobs(slug)
        for job in jobs:
            title = job.get("text", "")
            if terms and not title_matches(title, terms):
                continue
            return job.get("hostedUrl", "") or ""
    if ats == "greenhouse":
        jobs = fetch_greenhouse_jobs(slug)
        for job in jobs:
            title = job.get("title", "")
            if terms and not title_matches(title, terms):
                continue
            return job.get("absolute_url", "") or ""
    if ats == "smartrecruiters":
        jobs = fetch_smartrecruiters_jobs(slug)
        for job in jobs:
            title = job.get("name", "")
            if terms and not title_matches(title, terms):
                continue
            return job.get("ref", "") or job.get("applyUrl", "") or ""
    return ""


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


def rank_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = []
    for item in items:
        summary = f"{item.get('title','')} {item.get('summary','')}".lower()
        item["score"] = 2 if "ai" in summary else 0
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
            "You are a career strategist for industry roles.",
            "Given a user profile and a single news item, explain why it matters.",
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


def analyze_resume(agent: Any, resume_text: str, inputs: Profile, company_count: int) -> Dict[str, Any]:
    if not agent or not resume_text.strip():
        return {}
    prompt = (
        "Extract structured insights from this resume and optional inputs.\n"
        "Return JSON only with keys: inferred_industry, inferred_role, strengths, "
        "strategy, keywords, suggested_companies.\n"
        "strengths: 4-6 bullets. strategy: 3-5 bullets tied to industry trends.\n"
        "keywords: 8-12 short phrases.\n"
        f"suggested_companies: list of {company_count} objects with name, category, notes, ats, ats_slug.\n"
        "ats must be one of: lever, greenhouse, smartrecruiters.\n"
        "ats_slug is the company slug for that ATS.\n"
        "Only include real, currently operating companies. Do not invent names.\n\n"
        f"Inputs:\n"
        f"- Target role: {inputs.role}\n"
        f"- Industry: {inputs.industry}\n"
        f"- Skills: {', '.join(inputs.skills)}\n"
        f"- Location: {inputs.location}\n"
        f"- Seniority: {inputs.seniority}\n"
        f"- Keywords: {', '.join(inputs.keywords)}\n\n"
        f"Resume:\n{resume_text}\n"
    )
    try:
        response = agent.run(prompt)
        text = getattr(response, "content", None) or getattr(response, "output", None)
        return parse_json_block(text or str(response))
    except Exception:
        return {}




st.set_page_config(page_title="Resume Strategy Agent", layout="wide")
st.title("Resume Strategy Agent")

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
    company_count = st.number_input("Number of company targets", min_value=10, max_value=40, value=20, step=5)
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
    return rank_items(items)


if st.button("Run analysis"):
    resume_text = ""
    if resume_file is not None:
        resume_text = extract_pdf_text(resume_file.getvalue())
    if not resume_text:
        st.warning("Upload a resume to continue.")
        st.stop()

    agent = make_agent(model_id) if use_agent else None
    resume_insights = analyze_resume(agent, resume_text, profile, int(company_count))
    inferred_role = resume_insights.get("inferred_role", "")
    inferred_industry = resume_insights.get("inferred_industry", "")
    resume_keywords = resume_insights.get("keywords", [])
    strengths = resume_insights.get("strengths", [])
    strategy = resume_insights.get("strategy", [])
    suggested_companies = resume_insights.get("suggested_companies", [])
    verified_companies = []
    for company in suggested_companies:
        name = company.get("name", "")
        job_url = find_job_url_from_ats(
            company,
            profile.role or inferred_role or "",
            resume_keywords,
        )
        if not job_url:
            continue
        company["job_url"] = job_url
        verified_companies.append(company)
    suggested_companies = verified_companies

    rss_urls = list(DEFAULT_NEWS_RSS)
    query_parts = [
        profile.industry or inferred_industry or "",
        inferred_role or "",
        " ".join(profile.keywords),
        " ".join(resume_keywords),
    ]
    news_query = " ".join([p for p in query_parts if p]).strip()
    if not news_query:
        st.warning("Add an industry, keywords, or upload a resume to fetch news.")
        st.stop()

    with st.spinner("Fetching news..."):
        news_items = collect_news(profile, rss_urls, news_query)

    if use_agent:
        if not agent:
            st.warning("Agno not available. Install requirements and restart.")
        else:
            with st.spinner("Explaining matches..."):
                enrich_with_agent(agent, profile, news_items[:10])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Strengths")
        if strengths:
            for item in strengths:
                st.write(f"- {item}")
        else:
            st.write("No strengths generated yet.")

        st.subheader("Strategy")
        if strategy:
            for item in strategy:
                st.write(f"- {item}")
        else:
            st.write("No strategy generated yet.")
        st.subheader("Industry News")
        for item in news_items[:15]:
            st.markdown(f"**{item.get('title','')}**")
            st.caption(f"{item.get('source','')} • {item.get('published','')}")
            if item.get("why"):
                st.write(item["why"])
            else:
                summary = item.get("summary", "")
                if summary:
                    st.write(summary)
            st.link_button("Open", item.get("url", ""))
            st.divider()

    with col2:
        st.subheader("Company Targets")
        if not suggested_companies:
            st.info("No company targets generated yet.")
        else:
            rows = []
            for company in suggested_companies:
                rows.append(
                    {
                        "Company": company.get("name", ""),
                        "Category": company.get("category", ""),
                        "Notes": company.get("notes", ""),
                        "Job": company.get("job_url", ""),
                    }
                )
            st.dataframe(rows, use_container_width=True)

else:
    st.info("Upload a resume and click Run analysis.")
