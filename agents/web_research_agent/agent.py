from pathlib import Path

# Importing these registers the "web_search" and "fetch_page" tools.
from tools import registry, web_fetch, web_search  # noqa: F401

SYSTEM_PROMPT = (Path(__file__).parent / "skills.md").read_text(encoding="utf-8")
TOOLS = registry.get_schemas(["web_search", "fetch_page"])
