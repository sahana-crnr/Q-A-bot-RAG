"""
extractor.py - Parses executives, designations, and direct LinkedIn profile links from HTML.
"""

import re
from typing import Optional
from bs4 import BeautifulSoup, Tag

# Target roles to identify and extract
TARGET_ROLES_REGEX = re.compile(
    r"\b("
    r"CEO|Chief Executive Officer|"
    r"CTO|Chief Technology Officer|"
    r"President|Co-President|"
    r"Vice President|VP|SVP|EVP|Senior Vice President|Executive Vice President|"
    r"Chief Operating Officer|COO|"
    r"Chief Financial Officer|CFO|"
    r"Chief Information Officer|CIO|"
    r"Chief Revenue Officer|CRO|"
    r"Chief Medical Officer|CMO|"
    r"Chief Strategy Officer|CSO|"
    r"Chief Innovation Officer|"
    r"Chief Product Officer|CPO|"
    r"Chief Outcomes Officer|"
    r"Chief Marketing Officer|"
    r"Chief People Officer|"
    r"Chief Privacy|Chief of Staff|"
    r"Executive Chairman|Chairman|"
    r"Board Member|Board of Directors|Director|"
    r"Founder|Co-Founder|Head of"
    r")\b",
    re.IGNORECASE,
)

ROLE_TERMS = {
    "ceo", "cto", "cfo", "coo", "cio", "cro", "cmo", "cso", "cpo",
    "officer", "president", "vice president", "vp", "svp", "evp",
    "founder", "co-founder", "director", "chairman", "board", "partner", "head of",
    "chief", "staff", "general counsel", "managing director"
}

STOP_WORDS = {
    "our team", "leadership team", "board of directors", "executive team",
    "meet the team", "about us", "who we are", "read more", "view profile",
    "contact us", "get in touch", "learn more", "careers", "advisors",
    "investors", "mission", "privacy policy", "terms of use", "our leadership",
    "b.well", "pdw", "clearjet", "twelve", "company", "team", "news",
    "linkedin", "view bio", "read bio", "profile", "more", "menu", "close"
}


NON_PERSON_WORDS = {
    "series", "round", "news", "latest", "airpower", "competencies", "overview",
    "press", "events", "solutions", "products", "services", "platform", "insights",
    "resources", "privacy", "terms", "cookie", "copyright", "rights", "reserved",
    "ventures", "capital", "technologies", "holdings", "group", "corporation",
    "partners", "inc", "llc", "corp", "ltd", "core", "mission", "vision",
    "about", "team", "leadership", "board", "directors", "executives", "bwell"
}


def clean_text(text: str) -> str:
    """Clean whitespace and formatting."""
    return re.sub(r"\s+", " ", text).strip()


def is_role_string(text: str) -> bool:
    """Check if a string represents a job title or executive role."""
    t = text.lower()
    return any(r in t for r in ROLE_TERMS) or bool(TARGET_ROLES_REGEX.search(text))


def is_valid_name(name: str) -> bool:
    """Validate whether a candidate string looks like an executive person name."""
    name = clean_text(name)
    if not name or len(name) < 3 or len(name) > 40:
        return False
    if name.lower() in STOP_WORDS:
        return False
    if is_role_string(name):
        return False
    words = [w for w in re.split(r"[\s\.,]+", name) if w]
    if len(words) < 2 or len(words) > 5:
        return False
    # Avoid single letter tokens at the end like "Series A"
    if len(words[-1]) < 2:
        return False
    if any(char.isdigit() for char in name):
        return False
    # Check if any word belongs to non-person business vocabulary
    for w in words:
        if w.lower() in NON_PERSON_WORDS:
            return False
    if not all(w[0].isupper() for w in words if w.isalpha()):
        return False
    return True


def parse_name_from_slug(lk_url: str) -> str:
    """Extract person name from LinkedIn URL slug (e.g. /in/kristen-valdes -> Kristen Valdes)."""
    slug = lk_url.split("/in/")[-1].split("?")[0].strip("/")
    parts = slug.split("-")
    ignore_tokens = {"mba", "phd", "cpa", "phr", "jd", "md", "bba", "ms", "bs"}
    clean_parts = [p.capitalize() for p in parts if p.isalpha() and p.lower() not in ignore_tokens]
    return " ".join(clean_parts[:3])


def categorize_role(title: str) -> str:
    """Map a detailed title into an executive category."""
    title_lower = title.lower()
    if "ceo" in title_lower or "chief executive" in title_lower:
        return "CEO"
    elif "cto" in title_lower or "chief technology" in title_lower:
        return "CTO"
    elif "president" in title_lower and "vice" not in title_lower and "vp" not in title_lower:
        return "President"
    elif any(k in title_lower for k in ["vice president", "vp", "svp", "evp"]):
        return "Vice President"
    elif any(k in title_lower for k in ["board", "director", "chairman"]):
        return "Board Member"
    elif "founder" in title_lower:
        return "Founder"
    elif any(k in title_lower for k in ["cfo", "coo", "cio", "cmo", "cso", "cpo", "chief"]):
        return "C-Suite"
    return "Executive"


def extract_executives_from_html(html_content: str, page_url: str = "") -> list[dict]:
    """
    Extract executive names, designations, and direct LinkedIn profile links from HTML.
    Returns a list of dicts: {name, title, category, linkedin_url, source_page, direct_source}.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    executives = []
    seen_links = set()
    seen_names = set()

    # Clean out scripts, styles, forms, footers, headers
    for tag in soup(["script", "style", "nav", "footer", "form", "noscript", "svg"]):
        tag.decompose()

    # Strategy 1: Direct LinkedIn Link Driven Extraction (High Precision & 0 API cost)
    linkedin_tags = soup.find_all("a", href=lambda h: h and "linkedin.com/in/" in h)
    for a in linkedin_tags:
        lk_url = a["href"].split("?")[0].rstrip("/")
        if lk_url in seen_links:
            continue
        seen_links.add(lk_url)

        # Walk up to find the highest ancestor that isolates this person (exactly 1 linkedin link)
        card = a
        curr = a.parent
        while curr and curr.name not in ["body", "html", "[document]"]:
            lk_count = len(curr.find_all("a", href=lambda h: h and "linkedin.com/in/" in h))
            if lk_count == 1:
                card = curr
                curr = curr.parent
            else:
                break

        lines = [re.sub(r"\s+", " ", s).strip() for s in card.stripped_strings if s.strip()]
        lines = [l for l in lines if l.lower() not in STOP_WORDS and len(l) <= 70]

        cand_name = None
        cand_title = None

        for l in lines:
            if is_role_string(l):
                if not cand_title:
                    cand_title = l
            else:
                words = l.split()
                if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                    if not cand_name:
                        cand_name = l

        if not cand_name:
            cand_name = parse_name_from_slug(lk_url)

        if not cand_title:
            cand_title = "Executive"

        norm_name = cand_name.lower()
        if norm_name not in seen_names:
            seen_names.add(norm_name)
            executives.append({
                "name": cand_name,
                "title": cand_title,
                "category": categorize_role(cand_title),
                "linkedin_url": lk_url,
                "source_page": page_url,
                "direct_source": True,
            })

    # Strategy 2: Card / Container Search (for pages without direct LinkedIn links)
    if not executives:
        containers = soup.find_all(["div", "article", "section", "li"])
        for container in containers:
            text = container.get_text(separator=" ", strip=True)
            role_match = TARGET_ROLES_REGEX.search(text)
            if not role_match:
                continue

            name_tag = container.find(["h2", "h3", "h4", "h5", "strong", "b"])
            if not name_tag:
                continue

            cand_name = clean_text(name_tag.get_text())
            if not is_valid_name(cand_name):
                continue

            title = ""
            for elem in container.find_all(["p", "span", "div", "h4", "h5", "h6"]):
                if elem == name_tag:
                    continue
                elem_text = clean_text(elem.get_text())
                if elem_text and is_role_string(elem_text) and elem_text != cand_name:
                    title = elem_text
                    break

            if not title:
                title = role_match.group(0)

            title = title[:100].strip()
            norm_name = cand_name.lower()
            if norm_name not in seen_names:
                seen_names.add(norm_name)
                executives.append({
                    "name": cand_name,
                    "title": title,
                    "category": categorize_role(title),
                    "linkedin_url": "",
                    "source_page": page_url,
                    "direct_source": False,
                })

    return executives
