# Resume Strategy Agent

Resume Strategy Agent helps job seekers turn a resume into a focused search plan. It analyzes your resume, suggests a target role, summarizes what’s happening in your industry, and surfaces a radar of relevant companies to target.

Use it to:
- Clarify what roles you’re best positioned for
- Build a targeted company list instead of spray‑and‑pray
- Stay current on industry trends that shape hiring

[screen-capture.webm](https://github.com/user-attachments/assets/5fda02db-4489-455f-bb0e-556cb0f920e8)

## Features
- PDF resume upload (in‑memory).
- Auto‑suggested target role from resume.
- Industry + location multi‑selects.
- Resume strengths + strategy tied to industry trends.
- Industry news summary + top links.
- Target company list derived from resume + industry discovery.

## Typical workflow
1) Upload your resume.  
2) Confirm or adjust the suggested target role.  
3) Pick industries and locations.  
4) Run analysis to get strengths, strategy, industry pulse, and company targets.

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
- Uses OpenAI via Agno (model id set in code).
- Uses an AI agent (Agno + OpenAI) for resume analysis and summarization.
