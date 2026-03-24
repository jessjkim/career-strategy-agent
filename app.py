import os
from dataclasses import dataclass
import io
import json
import re
from typing import List, Dict, Any
from urllib.parse import quote_plus

import feedparser
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
    note: str


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


def dedupe_company_list(companies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped = []
    for company in companies:
        name = company.get("name", "").strip()
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(company)
    return deduped


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
        "strengths: 4-6 bullets grounded in resume evidence.\n"
        "strategy: 3-5 bullets tied to industry trends and the user's background.\n"
        "Avoid generic phrasing; each bullet should include a concrete detail from the resume.\n"
        "keywords: 8-12 short phrases.\n"
        f"suggested_companies: list of {company_count} objects with name, category, notes.\n"
        "Only include real, currently operating companies. Do not invent names.\n\n"
        f"Inputs:\n"
        f"- Target role: {inputs.role}\n"
        f"- Industry: {inputs.industry}\n"
        f"- Skills: {', '.join(inputs.skills)}\n"
        f"- Location: {inputs.location}\n"
        f"- Seniority: {inputs.seniority}\n"
        f"- Keywords: {', '.join(inputs.keywords)}\n"
        f"- Note: {inputs.note}\n\n"
        f"Resume:\n{resume_text}\n"
    )
    try:
        response = agent.run(prompt)
        text = getattr(response, "content", None) or getattr(response, "output", None)
        return parse_json_block(text or str(response))
    except Exception:
        return {}


def extract_companies_from_items(agent: Any, items: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    if not agent or not items:
        return []
    sample = "\n".join(
        [f"- {item.get('title','')} | {item.get('source','')}" for item in items[:40]]
    )
    prompt = (
        "Extract company names mentioned or strongly implied by these search results.\n"
        f"Return JSON only: list of up to {limit} objects with name, category, notes.\n"
        "Only include real companies. Do not invent names.\n\n"
        f"Results:\n{sample}\n"
    )
    try:
        response = agent.run(prompt)
        text = getattr(response, "content", None) or getattr(response, "output", None)
        parsed = parse_json_block(text or str(response))
        if isinstance(parsed, list):
            return parsed
        return parsed.get("companies", [])
    except Exception:
        return []


def collect_company_search_items(industry: str, inferred_industry: str, keywords: List[str]) -> List[Dict[str, Any]]:
    seed = industry or inferred_industry
    if not seed:
        return []
    query_templates = [
        f"{seed} startup funding",
        f"{seed} Series A",
        f"{seed} Series B",
        f"{seed} raises funding",
        f"{seed} partnership",
        f"{seed} launches product",
        f"{seed} acquisition",
    ]
    if keywords:
        query_templates.extend([f"{seed} {kw} company" for kw in keywords[:5]])

    items = []
    for query in query_templates:
        items.extend(fetch_rss(build_google_news_rss(query)))
    return dedupe_items(items)


def summarize_news(agent: Any, news_items: List[Dict[str, Any]]) -> str:
    if not agent or not news_items:
        return ""
    sample = "\n".join(
        [f"- {item.get('title','')} | {item.get('source','')}" for item in news_items[:25]]
    )
    prompt = (
        "Summarize the key themes and signals in these headlines in 3-5 sentences.\n"
        "Focus on industry shifts, funding, regulation, and product adoption.\n"
        "Avoid listing headlines; synthesize trends.\n\n"
        f"Headlines:\n{sample}\n"
    )
    try:
        response = agent.run(prompt)
        text = getattr(response, "content", None) or getattr(response, "output", None)
        return (text or "").strip()
    except Exception:
        return ""


def infer_role(agent: Any, resume_text: str) -> str:
    if not agent or not resume_text.strip():
        return ""
    prompt = (
        "Infer the most likely target role from this resume. "
        "Return a short role title only.\n\n"
        f"Resume:\n{resume_text}\n"
    )
    try:
        response = agent.run(prompt)
        text = getattr(response, "content", None) or getattr(response, "output", None)
        return (text or "").strip().splitlines()[0][:80]
    except Exception:
        return ""





st.set_page_config(page_title="Resume Strategy Agent", layout="wide")

st.markdown(
    """
    <style>
    .app-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .app-subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 24px;
    }
    .cta-wrap {
        text-align: center;
        margin: 8px 0 18px;
    }
    .stepper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 18px;
        font-weight: 600;
        color: #4b5563;
        margin: 14px 0 4px;
    }
    .step {
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }
    .step-badge {
        width: 26px;
        height: 26px;
        border-radius: 999px;
        background: #2f6fed;
        color: #ffffff;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 14px;
        font-weight: 700;
    }
    .step-divider {
        width: 36px;
        height: 1px;
        background: #d1d5db;
    }
    .input-title {
        font-size: 18px;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 10px;
    }
    .input-badge {
        width: 22px;
        height: 22px;
        border-radius: 999px;
        background: #dbeafe;
        color: #1d4ed8;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: 700;
    }
    .badge-success {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #e9f5e6;
        color: #1f7a1f;
        border: 1px solid #b9e0b2;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 14px;
        font-weight: 600;
    }
    .tab-card {
        background: #f5f5f0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
        border-left: 5px solid #2f6fed;
    }
    .tab-card.green {
        border-left-color: #2e9b50;
    }
    .news-item {
        padding: 10px 0;
        border-bottom: 1px solid #ececec;
    }
    .news-title {
        font-weight: 600;
    }
    .news-meta {
        color: #6b7280;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">Resume strategy agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Upload your resume and get a personalized job search strategy</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="cta-wrap">We’ll analyze your strengths, build a target company radar, and summarize the most relevant industry signals.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="stepper">'
    '<span class="step"><span class="step-badge">1</span>Upload resume</span>'
    '<span class="step-divider"></span>'
    '<span class="step"><span class="step-badge">2</span>Fill in details</span>'
    '<span class="step-divider"></span>'
    '<span class="step"><span class="step-badge">3</span>Get your strategy</span>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="input-title"><span class="input-badge">1</span>Upload your resume</div>',
    unsafe_allow_html=True,
)
resume_file = st.file_uploader("Upload resume (PDF)", type=["pdf"])
if resume_file is not None:
    st.markdown(
        f'<span class="badge-success">✓ {resume_file.name}</span>',
        unsafe_allow_html=True,
    )

resume_text = ""
if resume_file is not None:
    resume_text = extract_pdf_text(resume_file.getvalue())

model_id = "gpt-4o-mini"
agent = make_agent(model_id)

if resume_text and agent and st.session_state.get("resume_name") != resume_file.name:
    st.session_state.suggested_role = infer_role(agent, resume_text)
    st.session_state.resume_name = resume_file.name
    if st.session_state.suggested_role:
        st.session_state["target_role"] = st.session_state.suggested_role

st.markdown(
    '<div class="input-title"><span class="input-badge">2</span>What are you looking for?</div>',
    unsafe_allow_html=True,
)
inputs_disabled = resume_file is None
col_a, col_b, col_c = st.columns(3)
with col_a:
    role = st.text_input(
        "Target role",
        key="target_role",
        disabled=inputs_disabled,
    )
with col_b:
    industry_options = [
        "Healthcare",
        "Healthtech",
        "Fintech",
        "SaaS",
        "AI/ML",
        "Biotech",
        "Insurance",
        "Payments",
        "Enterprise",
        "Consumer",
    ]
    industries = st.multiselect(
        "Industries",
        options=industry_options,
        key="industry",
        disabled=inputs_disabled,
    )
with col_c:
    location_options = [
        "San Francisco, CA",
        "New York, NY",
        "Los Angeles, CA",
        "Seattle, WA",
        "Austin, TX",
        "Remote",
    ]
    locations = st.multiselect(
        "Locations",
        options=location_options,
        key="location",
        disabled=inputs_disabled,
    )
run_analysis = st.button("Run analysis", use_container_width=True, disabled=inputs_disabled)
st.divider()


profile = Profile(
    role=role.strip(),
    industry=", ".join([s.strip() for s in industries if s.strip()]),
    skills=[],
    location=", ".join([s.strip() for s in locations if s.strip()]),
    seniority="",
    keywords=[],
    exclusions=[],
    note="",
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


if run_analysis:
    if not resume_text:
        st.warning("Upload a resume to continue.")
        st.stop()

    company_count = 30
    if not agent:
        st.warning("Agno not available. Install requirements and restart.")
        st.stop()
    resume_insights = analyze_resume(agent, resume_text, profile, company_count)
    inferred_role = resume_insights.get("inferred_role", "")
    inferred_industry = resume_insights.get("inferred_industry", "")
    resume_keywords = resume_insights.get("keywords", [])
    strengths = resume_insights.get("strengths", [])
    strategy = resume_insights.get("strategy", [])
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

    suggested_companies = resume_insights.get("suggested_companies", [])
    company_search_items = collect_company_search_items(
        profile.industry,
        inferred_industry,
        resume_keywords,
    )
    discovered_companies = extract_companies_from_items(agent, company_search_items, int(company_count))
    suggested_companies = dedupe_company_list(suggested_companies + discovered_companies)

    news_summary = summarize_news(agent, news_items)

    with st.spinner("Explaining matches..."):
        enrich_with_agent(agent, profile, news_items[:10])

    st.header("Analysis")
    tabs = st.tabs(["Strengths", "Strategy", "Industry news", "Target companies"])

    with tabs[0]:
        if strengths:
            for item in strengths:
                st.markdown(
                    f'<div class="tab-card">{item}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No strengths generated yet.")

    with tabs[1]:
        if strategy:
            for item in strategy:
                st.markdown(
                    f'<div class="tab-card green">{item}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No strategy generated yet.")

    with tabs[2]:
        if news_summary:
            st.write(news_summary)
        else:
            st.info("No summary available.")
        st.markdown("### Top links")
        for item in news_items[:8]:
            st.markdown(
                f'<div class="news-item"><div class="news-title"><a href="{item.get("url","")}" target="_blank">{item.get("title","")}</a></div>'
                f'<div class="news-meta">{item.get("source","")} • {item.get("published","")}</div></div>',
                unsafe_allow_html=True,
            )

    with tabs[3]:
        if not suggested_companies:
            st.info("No company targets generated yet.")
        else:
            rows = []
            for company in suggested_companies:
                rows.append(
                    {
                        "Company": company.get("name", ""),
                        "Category": company.get("category", ""),
                        "Why it fits": company.get("notes", ""),
                    }
                )
            row_height = 36
            table_height = max(200, (len(rows) + 1) * row_height)
            st.dataframe(rows, use_container_width=True, height=table_height)


else:
    st.info("Upload a resume and click Run analysis.")
