import os, json, re
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from tools import TOOLS, validate_and_run

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

_client = None
# Language policy applied to EVERY LLM call (system instruction).
SYSTEM_INSTRUCTION = """You are a professional AI Data Scientist assistant for business users.
Language policy (mandatory):
- Always respond in clear, professional English only.
- Users may write in Hindi, Hinglish (Hindi written in Latin script) or any other language. Understand the request, but always answer in English.
- Never output Hindi (Devanagari) text or Hinglish words or sentences, even if the user asks for them.
- Keep column names, table names and data values exactly as they appear in the data.
- Use a concise, neutral business tone. Base every statement only on the data or context provided; never invent numbers."""

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")


def ask_llm(prompt: str, enforce_english: bool = True) -> str:
    global _client

    key = os.getenv("GROQ_API_KEY")

    if not key:
        raise RuntimeError("GROQ_API_KEY was not found.")

    if _client is None:
        _client = Groq(api_key=key)

    def _call(text):
        response = _client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_INSTRUCTION
                },
                {
                    "role": "user",
                    "content": text
                }
            ]
        )

        return response.choices[0].message.content or ""

    out = _call(prompt)

    if enforce_english and _DEVANAGARI.search(out):
        out = _call(
            "Rewrite the following text in professional English only. "
            "Do not use Hindi or Hinglish.\n\n" + out
        )

    return out


def _schema(df):
    return "\n".join(f"- {c} ({t})" for c, t in df.dtypes.astype(str).items())


def answer_question(df, question):
    plan_prompt = f"""You are a data analysis router.
Dataset columns:
{_schema(df)}

Available tools: {list(TOOLS.keys())}
Tool args: group_col, value_col, date_col, column, limit, agg (sum/mean/count/max/min).

User question (may be in English, Hindi or Hinglish): {question}

Reply ONLY with JSON, no extra text. Example:
{{"tool": "top_n", "args": {{"group_col": "Product", "value_col": "Revenue", "limit": 5, "agg": "sum"}}}}
If question cannot be answered with these columns/tools:
{{"tool": "clarify", "message": "what you need from the user, written in English"}}"""

    raw = ask_llm(plan_prompt, enforce_english=False)  # JSON routing, not user-facing text
    match = re.search(r"\{.*\}", raw, re.S)
    if not match:
        return "I could not understand the question. Please try rephrasing it.", None
    try:
        plan = json.loads(match.group())
    except json.JSONDecodeError:
        return "I could not understand the question. Please try rephrasing it.", None

    if plan.get("tool") == "clarify":
        return plan.get("message", "Please provide more detail."), None
    try:
        result = validate_and_run(df, plan.get("tool"), plan.get("args", {}))
    except Exception as e:
        return f"This question could not be answered: {e}", None

    result_df = result if isinstance(result, pd.DataFrame) else pd.DataFrame([result])
    explain_prompt = f"""Question: {question}
Verified result (calculated by Python):
{result_df.head(30).to_string(index=False)}

Explain this result in 3-4 lines of clear, professional English, even if the question was written in Hindi or Hinglish. Use ONLY the numbers above. Do not invent anything."""
    return ask_llm(explain_prompt), result_df
