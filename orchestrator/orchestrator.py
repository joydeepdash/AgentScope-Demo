# main.py
# Simple AgentScope multi-agent project:
# User -> Search Agent -> Dev Agent
#
# What it does:
# 1. User enters a topic
# 2. Search agent gathers web info
# 3. Dev agent creates a homepage HTML file
# 4. HTML file is saved locally
#
# Requirements:
# pip install agentscope openai duckduckgo-search
#
# Set your API key:
# Windows:
#   set OPENAI_API_KEY=your_key
#
# Linux/Mac:
#   export OPENAI_API_KEY=your_key

from pathlib import Path
from duckduckgo_search import DDGS

from agentscope.agent import ReActAgent
from agentscope.message import Msg
from agentscope.tool import Toolkit
from agentscope.models import OpenAIChatWrapper
from agentscope import init


# =========================================================
# Initialize AgentScope
# =========================================================

init(
    model_configs=[
        {
            "config_name": "gpt",
            "model_type": "openai_chat",
            "model_name": "gpt-4.1-mini",
            "api_key": None,  # Uses OPENAI_API_KEY env variable
        }
    ]
)

model = OpenAIChatWrapper(config_name="gpt")


# =========================================================
# Tool: Web Search
# =========================================================

def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    Returns summarized search snippets.
    """
    results_text = []

    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=5)

        for idx, r in enumerate(results, start=1):
            results_text.append(
                f"""
Result {idx}
Title: {r.get('title')}
Snippet: {r.get('body')}
URL: {r.get('href')}
"""
            )

    return "\n".join(results_text)


toolkit = Toolkit()
toolkit.register_tool_function(web_search)


# =========================================================
# Search Agent
# =========================================================

search_agent = ReActAgent(
    name="SearchAgent",
    model=model,
    toolkit=toolkit,
    sys_prompt=(
        "You are a web research agent.\n"
        "Your job is to search the web for the user's topic.\n"
        "Collect concise but useful information.\n"
        "Return a clean research summary."
    ),
)


# =========================================================
# Dev Agent
# =========================================================

dev_agent = ReActAgent(
    name="DevAgent",
    model=model,
    sys_prompt=(
        "You are a frontend developer.\n"
        "You receive research information.\n"
        "Create a COMPLETE HTML homepage.\n"
        "The page must contain:\n"
        "- Hero title\n"
        "- About/Summary section\n"
        "- Main information section\n"
        "- Clean styling using embedded CSS\n"
        "- Modern responsive layout\n"
        "- Return ONLY raw HTML code.\n"
    ),
)


# =========================================================
# Main Flow
# =========================================================

topic = input("Enter a topic: ")

print("\n[Search Agent] Researching...\n")

search_response = search_agent(
    Msg(
        name="user",
        content=f"Research this topic: {topic}",
        role="user",
    )
)

research_data = search_response.content

print("\n[Dev Agent] Building homepage...\n")

dev_prompt = f"""
Create a homepage about this topic:

TOPIC:
{topic}

RESEARCH:
{research_data}

Requirements:
- Single HTML file
- Embedded CSS only
- Add About/Summary section
- Add key information section
- Make it visually clean
"""

dev_response = dev_agent(
    Msg(
        name="SearchAgent",
        content=dev_prompt,
        role="assistant",
    )
)

html_content = dev_response.content


# =========================================================
# Save HTML File
# =========================================================

safe_name = topic.lower().replace(" ", "_")
output_file = Path(f"{safe_name}_homepage.html")

output_file.write_text(html_content, encoding="utf-8")

print(f"\nHTML page saved successfully:")
print(output_file.resolve())
