"""
This script is designed to sequentially crawl through multiple URLs from the `sitemap.xml` 
and extract the relevant content from each page.
It uses the `crawl4ai` library to perform the crawling and content extraction.
The script is asynchronous and can handle multiple URLs concurrently.

This script will specifically crawl the ATCC website 
using the URLs in this sitemap: https://www.atcc.org/sitemap.xml
"""
import os
import requests
import asyncio

__location__ = os.path.dirname(os.path.abspath(__file__))
__base_dir__ = os.path.dirname(os.path.dirname(__location__))
__output__ = os.path.join(__base_dir__, 'data', 'raw', 'web_crawled')

from typing import List

from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

from xml.etree import ElementTree as ET

async def crawl_sequential(urls: List[str]):
    # Configure the crawler
    browser_config = BrowserConfig(
        headless=True,
        browser_type='chrome'
        )

    run_config = CrawlerRunConfig(
        word_count_threshold=10,  # Minimum word count for content extraction
        excluded_tags=['form', 'header', 'footer'],  # Tags to exclude from content extraction
        exclude_external_links=True,  # Exclude external links
        remove_overlay_elements=True,  # Remove overlay elements from content

        # Cache control
        cache_mode=CacheMode.ENABLED,  # Enable caching, if available
        markdown_generator=DefaultMarkdownGenerator()  # Use the default markdown generator
        )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        await crawler.start()  # Start the crawler
        
        try:
            session_id = "session1"  # Reuse same session for all URLs
            
            for url in urls:
                result = await crawler.arun(
                    url=url,
                    config=run_config,
                    session_id=session_id  # Reuse the session ID
                )
                if result.success:
                    print(f"Successfully crawled {url}")
                    print(f"Markdown length: {len(result.markdown.raw_markdown)}")
                else:
                    print(f"Failed to crawl {url}: {result.error_message}")
                    print(f"Status Code: {result.status_code}")
        finally:
            await crawler.close()  # Close the crawler and browser after all URLs are processed
    

def get_urls_from_sitemap(sitemap_url: str) -> List[str]:
    """
    Fetches all URLs from the given sitemap URL.

    Returns:
        List[str]: A list of extracted URLs.
    """
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()  # Raise an error for bad responses

        # Parse the XML content
        root = ET.fromstring(response.content)
        namespace = {'xmlns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        urls = [loc.text for loc in root.findall('.//xmlns:loc', namespace)]

        return urls
    except requests.RequestException as e:
        print(f"Error fetching sitemap: {e}")
        return []
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")
        return []

def check_url_relevance(url: str, source='atcc') -> bool:
    """
    Checks if the URL is relevant based on specific criteria.
    This function is used to filter URLs based on specific keywords or patterns.
    If the URL contains any of the keywords, it is considered relevant.
    If the URL is not relevant, it is excluded from the crawling process.

    Args:
        url (str): The URL to check.
    
    Returns:
        bool: True if the URL is relevant, False otherwise.

    Keywords to check for: (specific to the ATCC website)
    + "resources"
        + "application-notes"
        + "culture-guides"
        + "microbial-media-formulations"
        + "product-sheets"
        + "safety-data-sheets"
        + "technical-documents"
        + "white-papers"
    + "applications"
    + "cell-products"
    + "the-science"
        + "genetic-engineering"
        + "culturing-cells"
    """

    # Helper function to check if a keyword is in the URL
    def contains_keyword(text: str, keyword_list: list) -> bool:
        """
        Helper function to check if any keyword in the list is present in the text.

        Args:
            text (str): The text to check.
            keyword_list (list): A list of keywords to check for.

        Returns:
            bool: True if any keyword is found in the text, False otherwise.
        """
        for keyword in keyword_list:
            if keyword in text:
                return True
        return False

    if source=='atcc':
        atcc_keywords = [
            "resources/application-notes",
            "resources/culture-guides",
            "resources/microbial-media-formulations",
            "resources/product-sheets",
            "resources/safety-data-sheets",
            "resources/technical-documents",
            "resources/white-papers",
            "applications",
            "cell-products",
            "the-science/genetic-engineering",
            "the-science/culturing-cells"
        ]
        if contains_keyword(url, keyword_list=atcc_keywords):
            return True
        else:
            return False
    else:
        return True  # If not ATCC, consider all URLs relevant (can modify as needed)

async def main():
    urls = get_urls_from_sitemap("https://www.atcc.org/sitemap.xml")
    relevant_urls = [url for url in urls if check_url_relevance(url, source='atcc')]
    
    if relevant_urls:
        await crawl_sequential(relevant_urls)
    else:
        print("No relevant URLs found for crawling.")

if __name__ == "__main__":
    asyncio.run(main())