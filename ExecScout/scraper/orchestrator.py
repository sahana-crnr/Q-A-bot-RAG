"""
orchestrator.py - Coordinates crawling, extraction, SerpApi enrichment, and contact discovery.
"""

import logging
from typing import Callable, Optional
from urllib.parse import urlparse

from .crawler import discover_leadership_pages, fetch_html, get_base_domain, normalize_url
from .extractor import extract_executives_from_html
from .serp_enricher import SerpEnricher
from .contact_finder import discover_company_contacts, enrich_executives_with_contacts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_executive_pipeline(
    company_url: str,
    serpapi_key: Optional[str] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> tuple[list[dict], dict]:
    """
    Main pipeline to discover company executives, LinkedIn profiles, and contact details.

    Args:
        company_url: e.g. "https://www.icanbwell.com/"
        serpapi_key: Optional SerpApi key (conserves 250 quota)
        progress_callback: Optional callback func(status_message, progress_fraction)

    Returns:
        tuple: (list of executive dicts, stats dict)
    """
    def report(msg: str, frac: float):
        if progress_callback:
            progress_callback(msg, frac)
        logger.info(f"[{int(frac * 100)}%] {msg}")

    normalized_url = normalize_url(company_url)
    domain = get_base_domain(normalized_url)
    company_name = domain.split(".")[0].capitalize()

    enricher = SerpEnricher(api_key=serpapi_key)

    report(f"Crawling {domain} to discover leadership pages...", 0.15)
    candidate_urls = discover_leadership_pages(normalized_url)

    all_executives = []
    seen_names = set()
    homepage_html = ""

    report("Scanning candidate leadership pages for executives & LinkedIn links...", 0.35)
    for idx, page_url in enumerate(candidate_urls):
        html = fetch_html(page_url)
        if not html:
            continue
        if page_url == normalized_url:
            homepage_html = html

        execs = extract_executives_from_html(html, page_url)
        for e in execs:
            norm_name = e["name"].lower()
            if norm_name not in seen_names:
                seen_names.add(norm_name)
                e["company"] = company_name
                e["domain"] = domain
                all_executives.append(e)

    # If direct LinkedIn executives were found from dedicated leadership pages,
    # discard loose unlinked cards from marketing/home sections
    direct_execs = [e for e in all_executives if e.get("direct_source")]
    if direct_execs:
        all_executives = direct_execs

    # If no executives found on static pages (e.g. Twelve / ClearJet JS sites), use 1 SerpApi query fallback
    if not all_executives and serpapi_key:
        report("No static leadership page found. Querying key executives via SerpApi...", 0.55)
        batch_execs = enricher.search_company_executives(company_name, domain)
        for e in batch_execs:
            norm_name = e["name"].lower()
            if norm_name not in seen_names:
                seen_names.add(norm_name)
                e["company"] = company_name
                e["domain"] = domain
                all_executives.append(e)

    # Targeted SerpApi LinkedIn enrichment for executives missing a LinkedIn profile
    direct_linkedin_count = sum(1 for e in all_executives if e.get("linkedin_url"))
    report(f"Found {len(all_executives)} executives ({direct_linkedin_count} direct LinkedIn profiles)...", 0.70)

    for e in all_executives:
        if not e.get("linkedin_url") and serpapi_key:
            report(f"Looking up LinkedIn for {e['name']} via SerpApi...", 0.80)
            found_link = enricher.find_linkedin_for_executive(e["name"], company_name)
            if found_link:
                e["linkedin_url"] = found_link
                e["direct_source"] = False

    # Contact discovery
    report("Extracting corporate contact info & email patterns...", 0.90)
    company_contacts = {}
    if homepage_html:
        company_contacts = discover_company_contacts(homepage_html, domain)

    all_executives = enrich_executives_with_contacts(all_executives, domain, company_contacts)

    stats = enricher.get_stats()
    stats["total_executives"] = len(all_executives)
    stats["direct_linkedin_count"] = sum(1 for e in all_executives if e.get("linkedin_url") and e.get("direct_source"))
    stats["serpapi_enriched_count"] = sum(1 for e in all_executives if e.get("linkedin_url") and not e.get("direct_source"))
    stats["candidate_pages_scanned"] = len(candidate_urls)
    stats["domain"] = domain
    stats["company_name"] = company_name

    report("Complete! Executive intelligence extracted.", 1.0)
    return all_executives, stats
