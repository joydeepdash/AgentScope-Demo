#the orchestrator file contains a web search tool using Tavily and 4 agents: StructureAgent -> ContentAgent -> BuilderAgent -> DeployAgent
#tested and works on Python 3.11.9

#IMPORTS

#needed for web search tool
from tavily import AsyncTavilyClient
from agentscope.tool import ToolResponse
from agentscope.message import TextBlock
import asyncio
from dotenv import load_dotenv
load_dotenv()

#needed for first three agents
from agentscope.agent import ReActAgent
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter
from agentscope.tool import Toolkit
from agentscope.message import Msg

#needed for DeployAgent, these are standard imports
import os
import re
import socket
import http.server
import threading
import webbrowser


#needed for event loop 
import sys

# =========================================================
# MODEL
# =========================================================

model = OpenAIChatModel(
    model_name="gpt-4.1-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)
formatter = OpenAIChatFormatter()

# =========================================================
# DEPLOYMENT AND SERVING LOCATIONS
# =========================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

OUTPUT_DIR = os.path.join(project_root, "output")
PORT = 8000


# =========================================================
# WEB SEARCH TOOL
# =========================================================

async def web_search(query: str, max_results: int = 5) -> ToolResponse:
    """Search the web for current information on a topic using Tavily.

    Args:
        query (`str`):
            The search query string.
        max_results (`int`, defaults to `5`):
            Maximum number of search results to return.

    Returns:
        `ToolResponse`:
            A tool response whose text content is a numbered list of
            results (title, URL, and a short snippet for each), or an
            error message if the search failed.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text="Error: TAVILY_API_KEY is not set; cannot search.",
            )],
        )

    try:
        client = AsyncTavilyClient(api_key=api_key)
        result = await client.search(
            query=query,
            max_results=max_results,
            include_answer=False,
        )
    except Exception as e:
        return ToolResponse(
            content=[TextBlock(
                type="text",
                text=f"Error: web search failed for {query!r}: {e}",
            )],
        )

    results = result.get("results", [])
    if not results:
        text = f"No results found for {query!r}."
    else:
        lines = [f"Search results for {query!r}:"]
        for i, item in enumerate(results, start=1):
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            snippet = (item.get("content") or "").strip()
            if len(snippet) > 500:
                snippet = snippet[:500] + "..."
            lines.append(f"{i}. {title}\n   URL: {url}\n   {snippet}")
        text = "\n".join(lines)

    return ToolResponse(content=[TextBlock(type="text", text=text)])

toolkit = Toolkit()
toolkit.register_tool_function(web_search)

# =========================================================
# STRUCTURE AGENT
# =========================================================

structure_agent = ReActAgent(
    name="StructureAgent",
    model=model,
    toolkit=toolkit,
    formatter=formatter,
    sys_prompt="""
You are a website structure planning agent.

Your task:
1. Analyze the input topic.
2. Determine the most appropriate website structure.
3. Break the website into logical sections and subsections.
4. Define the purpose of each section.
5. Return only a structured outline that can be used by downstream agents.
6. Do not generate the final content for the sections.
7. Do not generate HTML, CSS, or JavaScript.
8. Focus only on information architecture and page organization.

Output format:

Website Title: <title>

Sections:
1. <section name>
   - Purpose: <purpose>

2. <section name>
   - Purpose: <purpose>

3. <section name>
   - Purpose: <purpose>

...
""",
)

# =========================================================
# CONTENT AGENT
# =========================================================

content_agent = ReActAgent(
    name="ContentAgent",
    model=model,
    toolkit=toolkit,
    formatter=formatter,
    sys_prompt="""
You are a website content generation agent.

Your sole responsibility is to generate high-quality content for each website section provided by the Structure Agent.

Guidelines:
1. Use the website structure provided as input.
2. Generate content for every section and subsection.
3. Ensure the content aligns with the user's original request and intended audience.
4. Write clear, engaging, and professional content.
5. Maintain a consistent tone and style throughout the website.
6. Include appropriate headings, subheadings, paragraphs, lists, and call-to-action text where relevant.
7. If information is not provided by the user, create reasonable placeholder content that fits the context.
8. Do not generate HTML, CSS, JavaScript, or any other code.
9. Do not modify the structure received from the Structure Agent.
10. Do not explain your reasoning.

Output Format:

Website Title: <title>

Section: <section name>

Heading:
<heading text>

Content:
<section content>

Subsections:

Subsection: <subsection name>
Content:
<subsection content>

---

Repeat for every section and subsection defined in the structure.
""",
)

# =========================================================
# BUILDER AGENT
# =========================================================

#remove line 10 from the sys_prompt if too much content
builder_agent = ReActAgent(
    name="BuilderAgent",
    model=model,
    formatter=formatter,
    sys_prompt="""
You are a website builder agent.

Your sole responsibility is to convert the website structure and content provided by upstream agents into a complete, production-ready HTML document.

Guidelines:
1. Use the provided structure and content exactly as input.
2. Generate a single, valid HTML5 document.
3. Include all required HTML elements (<html>, <head>, <body>, etc.).
4. Create a clean, modern, and responsive layout.
5. Use semantic HTML elements where appropriate (header, nav, main, section, article, footer, etc.).
6. Include embedded CSS within a <style> tag.
7. Include embedded JavaScript within a <script> tag only when necessary.
8. Ensure the website is visually appealing and easy to navigate.
9. Generate appropriate navigation links if necessary for all major sections.
10. Preserve all content generated by the Content Agent.
11. Do not invent new sections unless required for usability.
12. Ensure the HTML can be saved directly as an .html file and opened in a browser without modification.
13. Do not perform web searches or external lookups.
14. Do not explain your reasoning.
15. Do not include markdown formatting, code fences, or commentary.
16. Use the year 2026 for copyright tags on self-content (if someone else's content says otherwise do not refer to it).

Output Requirements:
- Return ONLY the complete HTML document.
- The output must begin with <!DOCTYPE html>.
- The output must end with </html>.
- No text is allowed before or after the HTML document.
""",
)

# =========================================================
# DEPLOY AGENT
# =========================================================

# this is a decoy agent (function) because its task is deterministic.
# it takes a Msg object (like a real agent would) so the event loop stays
# correct even if we later swap it for a real agent.
async def deploy_agent( #async used to make code refactoring easier in case we swap for a real agent
    msg: Msg,
    port: int = PORT,
    output_dir: str = OUTPUT_DIR,
    open_browser: bool = True,
) -> str:
    """Write the HTML from a Msg to disk, serve it locally, open a browser.

    Args:
        msg: The BuilderAgent's output Msg; its text content is the HTML.
        port: Preferred port; falls back to the next free one if taken.
        output_dir: Directory to write the file into (created if missing).
        open_browser: Whether to auto-open the URL in the default browser.

    Returns:
        The URL the site is being served at.
    """
    # --- extract the HTML string from the Msg object ---
    html = msg.get_text_content()
    if not html:
        raise ValueError(
            "deploy_agent: incoming Msg has no text content to deploy."
        )
    html = html.strip()

    # --- strip any accidental code fences the LLM may have added ---
    if html.startswith("```"):
        html = html.split("```", 2)[1]
        if html.startswith("html"):
            html = html[len("html"):]
        html = html.strip()

    if "<!DOCTYPE html>" not in html and "<html" not in html.lower():
        raise ValueError(
            "deploy_agent: content does not look like an HTML document; "
            "BuilderAgent may have returned prose or an empty string."
        )

    # --- derive a filename from the page <title>, fall back to 'site' ---
    title_match = re.search(
        r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL
    )
    raw_title = title_match.group(1).strip() if title_match else ""
    slug = re.sub(r"[^a-z0-9]+", "-", raw_title.lower()).strip("-")[:60]
    filename = f"{slug or 'site'}.html"

    # --- write the file to disk ---
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html)

    # --- find a free port (fallback if the preferred one is in use) ---
    def _free_port(preferred: int, attempts: int = 50) -> int:
        for p in range(preferred, preferred + attempts):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.bind(("127.0.0.1", p))
                    return p
                except OSError:
                    continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    chosen_port = _free_port(port)

    # --- serve the output directory (not the whole cwd) ---
    serve_dir = os.path.dirname(file_path)

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=serve_dir, **kwargs)

        def log_message(self, *args):  # silence request logging
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", chosen_port), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    url = f"http://localhost:{chosen_port}/{filename}"
    print(f"Site deployed at {url}")

    if open_browser:
        webbrowser.open(url)

    return url

async def main():

    if not os.getenv("OPENAI_API_KEY") or not os.getenv("TAVILY_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY and TAVILY_API_KEY must both be set.")
    
    user_input = input("> ")
    msg = Msg(
        name="user",
        role="user",
        content=user_input
    )
    try:
        msg = await structure_agent(msg)
        msg = await content_agent(msg)
        msg = await builder_agent(msg)
        url = await deploy_agent(msg) #await only used to keep event loop code same in case we use a real agent
    except Exception as e:
        sys.exit(f"Pipeline failed: {e}")
        
    print(url)
    print("Serving — press Ctrl+C to stop.")
    await asyncio.Event().wait()

if __name__ == "__main__":
    
    asyncio.run(main())


