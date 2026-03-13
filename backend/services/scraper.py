import httpx
from bs4 import BeautifulSoup

# Tags whose content should be stripped entirely (not just tag, but content too)
_NOISE_TAGS = ["script", "style", "nav", "header", "footer", "aside", "noscript", "iframe"]


def scrape_url(url: str) -> str:
    """Fetch a URL and return the main text content, stripping navigation/boilerplate."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        )
    }
    with httpx.Client(follow_redirects=True, timeout=15) as client:
        response = client.get(url, headers=headers)
        response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    # Prefer article / main content blocks if present
    main = soup.find("main") or soup.find("article") or soup.find(id="job-details") or soup.body
    if main is None:
        return soup.get_text(separator="\n", strip=True)

    lines = [line.strip() for line in main.get_text(separator="\n").splitlines()]
    # Remove empty lines and deduplicate consecutive blanks
    cleaned: list[str] = []
    for line in lines:
        if line:
            cleaned.append(line)
        elif cleaned and cleaned[-1] != "":
            cleaned.append("")

    return "\n".join(cleaned).strip()
