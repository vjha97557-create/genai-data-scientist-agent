# GenAI Data Scientist Agent

Excel/CSV upload -> cleaning -> Power BI-style dashboard -> insights -> AI Q&A -> SQL -> PDF Q&A -> forecast -> report (MD/HTML/PDF).

## Run (5 minute)
```
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # Mac/Linux: cp .env.example .env   (then add GEMINI_API_KEY to .env)
streamlit run app.py
```
The Dashboard, Data Profile, Chart Builder, Insights, Report, Forecast and manual SQL work without an API key.
Only "Ask AI", "Text-to-SQL" and "PDF Q&A" require a Groq API key.

## Tests
`python -m pytest -q`

## Environment variables (names only)
GROQ_API_KEY, Groq_MODEL (optional, default openai-model-name)

## Known limitations
- The forecast is a linear trend model (no seasonality).


## Project Team Members
      1.Vikash Jha
      2.Prateek Vats
      3.Harsh Mishra
      4.Adit Shaiwal
- PDF Q&A uses TF-IDF (embeddings would improve retrieval).
- SQLite demo database; use MySQL with a read-only user in production.
