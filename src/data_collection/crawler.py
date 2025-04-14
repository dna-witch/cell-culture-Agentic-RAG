# crawler.py

import asyncio
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig, CacheMode

async def main():
    # Configure the crawler
    browser_config = BrowserConfig(
        # headless=True,  # Run in headless mode
        # browser_type='chrome',  # Use Chrome browser
        # # Add any other browser-specific configurations here
    )

    run_config = CrawlerRunConfig(
        # max_depth=3,  # Maximum depth of crawling
        # max_pages=100,  # Maximum number of pages to crawl
        # Add any other run-specific configurations here
        word_count_threshold=10,  # Minimum word count for content extraction
        excluded_tags=['form', 'header', 'footer'],  # Tags to exclude from content extraction
        exclude_external_links=True,  # Exclude external links
        remove_overlay_elements=True,  # Remove overlay elements from content

        # Cache control
        cache_mode=CacheMode.ENABLED  # Enable caching, if available
    )

    # Get URLS from the `url_list.txt` file
    with open('url_list.txt', 'r') as file:
        urls = file.readlines()
    urls = [url.strip() for url in urls if url.strip()]  # Remove empty lines and whitespace

    # # Create an instance of the crawler (for a single URL)
    # async with AsyncWebCrawler(config=browser_config) as crawler:
    #     result = await crawler.arun(
    #         url = "https://www.atcc.org/resources/culture-guides/animal-cell-culture-guide",
    #         config=run_config
    #     )
    #     if result.success:
    #         print("Crawling completed successfully.")
    #         print("Crawled URLs:", result.url)
    #         print(f"Content: {result.markdown[:500]}")  # Print the first 500 characters of the content
    #     else:
    #         print(f"Crawling failed: {result.error_message}")
    #         print(f"Status Code: {result.status_code}")

    # Create an instance of the crawler (for multiple URLs)
    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Loop through each URL and crawl it
        for url in urls:
            print(f"Crawling URL: {url}")
            # Run the crawler for each URL
            result = await crawler.arun(
                url=url,
                config=run_config
            )
            # Check the result of the crawling
            if result.success:
                print("Crawling completed successfully.")
                print("Crawled URLs:", result.url)
                print(f"Content: {result.markdown[:500]}")  # Print the first 500 characters of the content
            else:
                print(f"Crawling failed: {result.error_message}")
                print(f"Status Code: {result.status_code}")

if __name__ == "__main__":
    asyncio.run(main())