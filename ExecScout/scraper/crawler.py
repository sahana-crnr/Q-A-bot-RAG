"""
crawler.py - Discovers and fetches leadership & team pages for a company.
"""

import logging
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

LEADERSHIP_KEYWORDS = [
    "leadership",
    "team",
    "about",
    "board",
    "executive",
    "management",
    "people",
    "company",
    "directors",
]

COMMON_FALLBACK_PATHS = [
    "/leadership",
    "/team",
    "/about",
    "/about-us",
    "/our-team",
    "/company-leadership",
    "/board-of-directors",
    "/about-bwell/",
    "/company",
]


def normalize_url(url: str) -> str:
    """Ensure URL has scheme and is stripped."""
    url = url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    return url


def get_base_domain(url: str) -> str:
    """Extract root domain e.g. icanbwell.com."""
    parsed = urlparse(normalize_url(url))
    domain = parsed.netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def fetch_html(url: str, timeout: int = 12) -> str:
    """Fetch HTML content with headers and error handling."""
    normalized = normalize_url(url)
    try:
        resp = requests.get(normalized, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if resp.status_code == 200:
            return resp.text
        logger.warning(f"Fetch failed for {normalized} with status {resp.status_code}")
    except requests.exceptions.SSLError:
        try:
            resp = requests.get(normalized, headers=HEADERS, timeout=timeout, verify=False)
            if resp.status_code == 200:
                return resp.text
        except Exception as e:
            logger.error(f"SSL retry failed for {normalized}: {e}")
    except Exception as e:
        logger.error(f"Error fetching {normalized}: {e}")
    return ""


def discover_leadership_pages(company_url: str) -> list[str]:
    """
    Given a company URL, crawl the homepage and discover candidate leadership / team pages.
    Returns prioritized list of candidate URLs.
    """
    normalized_home = normalize_url(company_url)
    parsed_home = urlparse(normalized_home)
    base_domain = get_base_domain(company_url)

    candidates = []

    # 1. Fetch homepage
    home_html = fetch_html(normalized_home)
    if home_html:
        soup = BeautifulSoup(home_html, "html.parser")
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"].strip()
            if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
                continue

            # Resolve relative link
            full_url = urljoin(normalized_home, href)
            parsed_link = urlparse(full_url)

            # Must be same domain or subdomain
            if base_domain not in parsed_link.netloc.lower():
                continue

            path_lower = parsed_link.path.lower()
            text_lower = a_tag.get_text().strip().lower()

            # Check if path or link text matches leadership keywords
            if any(k in path_lower or k in text_lower for k in LEADERSHIP_KEYWORDS):
                clean_url = f"{parsed_link.scheme}://{parsed_link.netloc}{parsed_link.path}"
                if clean_url not in candidates:
                    candidates.append(clean_url)

    # 2. Add common fallback paths if not already discovered
    for path in COMMON_FALLBACK_PATHS:
        fallback_url = f"{parsed_home.scheme}://{parsed_home.netloc}{path}"
        if fallback_url not in candidates:
            candidates.append(fallback_url)

    # Put homepage as last fallback
    if normalized_home not in candidates:
        candidates.append(normalized_home)

    logger.info(f"Discovered {len(candidates)} candidate leadership pages for {company_url}")
    return candidates
