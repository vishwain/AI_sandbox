from ddgs import DDGS

from tools.registry import register_tool

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the public web via DuckDuckGo. Returns a numbered list of "
            "results (title, URL, snippet) for the query. Use this to find "
            "current or fact-checkable information not already known."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query."},
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 5).",
                },
            },
            "required": ["query"],
        },
    },
}


def web_search(query, max_results=5):
    """Run a DuckDuckGo text search and format the results as a numbered list."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))

    if not results:
        return "No results found."

    lines = []
    for i, result in enumerate(results, start=1):
        title = result.get("title", "")
        href = result.get("href", "")
        body = result.get("body", "")
        lines.append(f"{i}. {title}\n   URL: {href}\n   {body}")
    return "\n".join(lines)


register_tool("web_search", _SCHEMA, web_search)
