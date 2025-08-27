#!/usr/bin/env python3
"""
CrowdWisdom — Full CrewAI pipeline single-file:
  - search (Tavily / Serper)
  - summary (<=500 words)
  - formatting (embed 2 images)
  - translations (Arabic, Hindi, Hebrew)
  - send to Telegram (English)

Requirements:
  pip install crewai crewai-tools litellm pydantic requests python-dotenv

Env:
  TAVILY_API_KEY or SERPER_API_KEY
  OPENAI_API_KEY or GEMINI_API_KEY or ANTHROPIC_API_KEY or GROQ_API_KEY
  TELEGRAM_BOT_TOKEN
  TELEGRAM_CHAT_ID

Run:
  python crowdwisdom_full_pipeline.py --post true --model gpt-4o-mini
"""

import os, re, json, time, argparse, requests
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from litellm import completion
from pydantic import BaseModel, Field

load_dotenv()

# Env
tavily_key = os.getenv("TAVILY_API_KEY")
telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")


BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {
    "chat_id": CHAT_ID,
    "text": "🚀 Test message from CrowdWisdom pipeline"
}

resp = requests.post(url, json=payload)
print(resp.json())


# CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process, TaskOutput
except Exception as e:
    raise SystemExit("CrewAI not installed. Run: pip install crewai") from e

try:
    from crewai_tools import TavilySearchTool, ScrapeWebsiteTool
except Exception:
    TavilySearchTool, ScrapeWebsiteTool = None, None

# -----------------------------
# LiteLLM helper
# -----------------------------
def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Universal call to LiteLLM with fallback to Groq"""
    try:
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"[Error] {model} failed: {e}, trying Groq fallback...")
        try:
            resp = completion(
                model="groq/llama-3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
            )
            return resp["choices"][0]["message"]["content"]
        except Exception as e2:
            print(f"[Fatal] Fallback also failed: {e2}")
            return "LLM request failed"

# -----------------------------
# Data Schemas
# -----------------------------
class NewsItem(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None
    image_url: Optional[str] = None
    source: Optional[str] = None
    published: Optional[str] = None

class SearchOutput(BaseModel):
    query: str
    window: str
    items: List[NewsItem] = Field(default_factory=list)

class SummaryOutput(BaseModel):
    markdown_summary: str
    tickers_mentioned: List[str] = Field(default_factory=list)

class FormattedOutput(BaseModel):
    markdown_with_images: str
    image_urls: List[str] = Field(default_factory=list)

class TranslatedOutput(BaseModel):
    translations: Dict[str, str]

# -----------------------------
# Utils
# -----------------------------
def word_count(txt: str) -> int:
    return len(re.findall(r"\b\w+\b", txt or ""))

def now_ist() -> str:
    return time.strftime("%Y-%m-%d %H:%M IST", time.localtime())

# -----------------------------
# Guardrails
# -----------------------------
def guard_search(output: TaskOutput) -> Tuple[bool, Any]:
    try:
        raw = output.json_dict or output.pydantic or output.raw
        parsed = (
            raw if isinstance(raw, SearchOutput)
            else SearchOutput(**(raw if isinstance(raw, dict) else json.loads(raw)))
        )
        if not parsed.items:
            return False, "No search results"
        return True, parsed.dict()
    except Exception as e:
        return False, f"search_guard_error: {e}"

def guard_summary(output: TaskOutput) -> Tuple[bool, Any]:
    try:
        raw = output.json_dict or output.pydantic or {"markdown_summary": output.raw, "tickers_mentioned": []}
        parsed = (
            raw if isinstance(raw, SummaryOutput)
            else SummaryOutput(**(raw if isinstance(raw, dict) else json.loads(raw)))
        )
        wc = word_count(parsed.markdown_summary)
        if wc > 500:
            return False, f"summary too long: {wc}"
        if not re.search(r"\b(S&P|Dow|Nasdaq|Treasury|Fed|yield|futures|earnings|volatility|index)\b", parsed.markdown_summary, re.I):
            return False, "summary may not be finance-focused"
        return True, parsed.dict()
    except Exception as e:
        return False, f"summary_guard_error: {e}"

def guard_format(output: TaskOutput) -> Tuple[bool, Any]:
    try:
        raw = output.json_dict or output.pydantic or {"markdown_with_images": output.raw, "image_urls": []}
        parsed = (
            raw if isinstance(raw, FormattedOutput)
            else FormattedOutput(**(raw if isinstance(raw, dict) else json.loads(raw)))
        )
        if len(parsed.image_urls) != 2:
            return False, "must include exactly 2 image URLs"
        if parsed.markdown_with_images.count("![") < 2:
            return False, "markdown must embed two images"
        return True, parsed.dict()
    except Exception as e:
        return False, f"format_guard_error: {e}"

def guard_translate(output: TaskOutput) -> Tuple[bool, Any]:
    try:
        raw = output.json_dict or output.pydantic or output.raw
        parsed = (
            raw if isinstance(raw, TranslatedOutput)
            else TranslatedOutput(**(raw if isinstance(raw, dict) else json.loads(raw)))
        )
        required = {"english","arabic","hindi","hebrew"}
        if set(parsed.translations.keys()) != required:
            return False, f"translations must include {sorted(required)}"
        return True, parsed.dict()
    except Exception as e:
        return False, f"translate_guard_error: {e}"

# -----------------------------
# Tools
# -----------------------------
search_tool = TavilySearchTool() if TavilySearchTool else None
scrape_tool = ScrapeWebsiteTool() if ScrapeWebsiteTool else None

# -----------------------------
# Agents
# -----------------------------
search_agent = Agent(
    role="Market News Researcher",
    goal="Find last-hour US market-close news.",
    backstory="Desk researcher for market wraps.",
    llm=None,  # LLM via call_llm
    tools=[t for t in (search_tool, scrape_tool) if t],
)

summary_agent = Agent(
    role="Market Wrap Writer",
    goal="Write crisp <=500 word markdown summary.",
    backstory="Sell-side strategist.",
    llm=None,
)

formatting_agent = Agent(
    role="Visual Formatter",
    goal="Embed 2 related images logically.",
    backstory="Visual editor.",
    llm=None,
)

translator_agent = Agent(
    role="Multilingual Translator",
    goal="Translate summary into Arabic, Hindi, Hebrew.",
    backstory="Financial translator.",
    llm=None,
)

sender_agent = Agent(
    role="Telegram Publisher",
    goal="Send markdown to Telegram.",
    backstory="Delivery agent.",
    llm=None,
)

# -----------------------------
# Tasks
# -----------------------------
search_task = Task(
    description="Search web for US market close news. Return SearchOutput.",
    agent=search_agent,
    expected_output="News headlines + links",
    tools=[t for t in (search_tool, scrape_tool) if t],
    output_pydantic=SearchOutput,
    guardrail=guard_search,
)

summary_task = Task(
    description="Produce markdown summary <=500 words with tickers.",
    agent=summary_agent,
    expected_output="Financial market summary",
    context=[search_task],
    output_pydantic=SummaryOutput,
    guardrail=guard_summary,
)

format_task = Task(
    description="Add 2 images to markdown summary.",
    agent=formatting_agent,
    expected_output="Markdown + 2 images",
    context=[summary_task, search_task],
    output_pydantic=FormattedOutput,
    guardrail=guard_format,
)

translate_task = Task(
    description="Translate into Arabic, Hindi, Hebrew.",
    agent=translator_agent,
    expected_output="Translated summaries",
    context=[format_task],
    output_pydantic=TranslatedOutput,
    guardrail=guard_translate,
)

send_task = Task(
    description="Send English summary to Telegram.",
    agent=sender_agent,
    expected_output="Confirmation message that the summary was successfully posted to Telegram.",
)

crew = Crew(
    agents=[search_agent, summary_agent, formatting_agent, translator_agent, sender_agent],
    tasks=[search_task, summary_task, format_task, translate_task, send_task],
    process=Process.sequential,
    verbose=1,
)

# -----------------------------
# Telegram Helper
# -----------------------------


def send_to_telegram(text: str):
    BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "Markdown"  # optional, allows formatting
    }

    resp = requests.post(url, json=payload)
    print("Telegram response:", resp.json())



# -----------------------------
# Runner
# -----------------------------
def run(post: bool = False, model: str = "gpt-4o-mini") -> Dict[str, Any]:
    print(f"[Info] Crew run started at {now_ist()} using model={model}")
    crew.kickoff()

    outputs = {
        "search": getattr(search_task, "output", None),
        "summary": getattr(summary_task, "output", None),
        "formatted": getattr(format_task, "output", None),
        "translations": getattr(translate_task, "output", None),
    }

    def to_json(obj):
        if not obj: return None
        try: return obj.json_dict or obj.pydantic or obj.raw
        except: return str(obj)

    results = {k: to_json(v) for k,v in outputs.items()}

    if post:
        eng = None
        translations = results.get("translations", {})
        if isinstance(translations, dict):
            eng = translations.get("translations", {}).get("english")
        if not eng:
            formatted = results.get("formatted", {})
            eng = formatted.get("markdown_with_images")
        if eng and telegram_token and telegram_chat_id:
            results["send"] = send_to_telegram(telegram_token, telegram_chat_id, eng)
        else:
            results["send"] = {"ok": False, "error": "no english text or missing env vars"}

    outdir = "run_artifacts"
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "outputs.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("[Done] Artifacts saved to run_artifacts/")
    return results

# -----------------------------
# CLI
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--post", default="false", help="true/false send to telegram")
    p.add_argument("--model", default="gpt-4o-mini", help="LLM model id")
    a = p.parse_args()
    res = run(post=a.post.lower() in ("1","true","yes"), model=a.model)
    print(json.dumps(res, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
