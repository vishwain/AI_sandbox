import requests
from bs4 import BeautifulSoup

from tools.registry import register_tool

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "fetch_page",
        "description": (
            "Fetch a URL and return its visible text content (scripts/styles "
            "stripped), truncated to max_chars. Use this to read the full "
            "body of a page found via web_search when the snippet isn't "
            "enough to answer the question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch."},
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum characters of text to return (default 4000).",
                },
            },
            "required": ["url"],
        },
    },
}

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; AI-Sandbox-WebResearchAgent/1.0)"}


def fetch_page(url, max_chars=4000):
    """Download *url* and return its stripped visible text, truncated to max_chars."""
    try:
        response = requests.get(url, headers=_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        return f"Error: could not fetch '{url}' ({exc})"

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    if len(text) > max_chars:
        text = text[:max_chars] + "..."
    return text or "No readable text found on page."


register_tool("fetch_page", _SCHEMA, fetch_page)
