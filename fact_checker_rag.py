from scrap import fetch_all_article_links, fetch_article_and_remove_code_blocks
from v_db import LlamaSearch
from v_db import WhooshSearcher


def main():
    urls = fetch_all_article_links("https://go.dev/blog/")
    for url in urls:
        text, sentences = fetch_article_and_remove_code_blocks(url)
        print(sentences)

if __name__ == "__main__":
    main()
