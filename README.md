# Resume Strategy Agent

Streamlit app that analyzes a resume, summarizes industry news, and produces a target company radar (no job scraping).

## Features
- PDF resume upload (in‑memory).
- Resume strengths + strategy tied to industry trends.
- Industry news summary + top links.
- Target company list derived from resume + news.

## Setup
```bash
cd /Users/jessicakim/Code/industry-news-agent
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run
```bash
python -m streamlit run app.py
```

## Configuration
Create a `.env` with:
```env
OPENAI_API_KEY=your_key_here
```

## Notes
- Uses OpenAI via Agno (model id configurable in the UI).
- News comes from Google News RSS queries built from the resume/inputs.
