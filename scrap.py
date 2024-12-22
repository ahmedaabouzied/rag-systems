from bs4 import BeautifulSoup
import requests
import re

# Function to fetch and clean HTML content
def fetch_and_clean_html(url):
    response = requests.get(url)
    response.raise_for_status()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title and main content
    title = soup.title.string if soup.title else "No Title"
    divs = soup.find("div", {"class": "Article"})
    if divs is not None:
        p_tags = divs.find_all("p")
        text = " ".join(p_tag.get_text() for p_tag in p_tags)

    return title, text

def fetch_article_and_remove_code_blocks(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        article = soup.find("div", {"class": "Article"})
        if not article:
            return None, None
        for code_block in article.find_all(["pre", "code"]):
            code_block.decompose()

        text = article.get_text(separator=" ")
        text = re.sub(r'\s+', ' ', text).strip()
        # Improved sentence splitting (handles abbreviations better)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty sentences
        return text, sentences
    except Exception as error:
        print(f"Error scrapping article from URL: {url}: {error}")
        return None, None

def fetch_all_article_links(url: str):
    response = requests.get(url)
    response.raise_for_status()

    # Parse with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title and main content
    title = soup.title.string if soup.title else "No Title"
    divs = soup.find("div", {"class": "Article"})
    links = []
    if divs is not None:
        for tag in divs.find_all("a", href=True):
            links.append(f"https://go.dev{tag["href"]}")

    return links
