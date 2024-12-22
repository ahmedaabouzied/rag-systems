from abc import ABC, abstractmethod
import asyncio

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import OrGroup, QueryParser
from whoosh.analysis import SimpleAnalyzer

import chromadb
import argparse
import getpass
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns

from langchain_text_splitters import TextSplitter
from scrap import fetch_all_article_links, fetch_and_clean_html

class Case(ABC):
    def get_vs(self):
        return self._vs

    def set_vs(self, vs: Chroma):
        self._vs = vs

    def get_embeddings(self):
        return self._embeddings

    def set_embeddings(self, v):
        self._embeddings = v

    def set_text_splitter(self, v: TextSplitter):
        self._text_splitter = v
    def get_text_splitter(self):
        return self._text_splitter

    @abstractmethod
    async def load_doc(self, url: str):
        _, doc = fetch_and_clean_html(url)
        splits = self.get_text_splitter().create_documents([doc])
        for split in splits:
            split.metadata["path"] = url
        self.get_vs().add_documents(documents=splits)
        print(f"Loaded doc from url {url}")

    @abstractmethod
    def search(self, term: str):
        res = self.get_vs().similarity_search_with_score(term, k=5)
        # for doc in res:
        #     print(f"""
        #             Score: {doc[1]}
        #             Doc: {doc[0].page_content[:120]}
        #         """)
        return [(doc[0].page_content, doc[0].metadata["path"], doc[1]) for doc in res]

class OpenAISearch(Case):
    def __init__(self):
        # Init open AI embedding
        if not os.environ.get("OPENAI_API_KEY"):
            os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
        openAI_embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large")
        self.set_embeddings(openAI_embeddings_model)

        persistent_client = chromadb.PersistentClient()
        vector_store = Chroma(
            client=persistent_client,
            collection_name="oai_search_collection",
            embedding_function=openAI_embeddings_model,
            # Set chroma to use cosine similarity instead of Squared L2
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.set_vs(vector_store)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False
        )
        self.set_text_splitter(text_splitter)

    async def load_doc(self, url:str):
        await super().load_doc(url)
    def search(self, term:str):
        return super().search(term)



class LlamaSearch(Case):
    def __init__(self):
        # Init open AI embedding
        embeddings = OllamaEmbeddings(model="llama3.2")
        self.set_embeddings(embeddings)

        persistent_client = chromadb.PersistentClient()
        vector_store = Chroma(
            client=persistent_client,
            collection_name="llama_search_collection",
            embedding_function=embeddings,
            # Set chroma to use cosine similarity instead of Squared L2
            collection_metadata={"hnsw:space": "cosine"}
        )
        self.set_vs(vector_store)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False
        )
        self.set_text_splitter(text_splitter)
    async def load_doc(self, url:str):
        try:
            await super().load_doc(url)
        except:
            print("Failed to load doc {url} into llama vector DB")
    def search(self, term:str):
        return super().search(term)

class WhooshSearcher(Case):
    def __init__(self, index_dir="whoosh_index"):
        self.index_dir = index_dir
        self.schema = Schema(
            content=TEXT(stored=True, analyzer=StemmingAnalyzer()),  # For full-text search
            path=ID(stored=True, unique=True),  # Unique identifier for each document
        )
        # Create or open the index
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
            self.index = create_in(self.index_dir, self.schema)
        else:
            self.index = open_dir(self.index_dir)

    async def load_doc(self, url:str):
        """
        Add a document to the index with its content and unique path.
        """
        try:
            _, doc = fetch_and_clean_html(url)
            writer = self.index.writer()
            writer.add_document(content=doc, path=url)
            writer.commit()
        except:
            print(f"Failed to load doc {url} into Whoosh searcher index")

    def preprocess_query(self, term: str):
        """
        Preprocess the query to remove stopwords and extract keywords.
        """
        analyzer = SimpleAnalyzer()
        tokens = [token.text for token in analyzer(term)]
        return " OR ".join(tokens)  # Join tokens with OR for flexible searching

    def search(self, term: str):
        """
        Perform a full-text search across all document content.
        """
        with self.index.searcher() as searcher:
            # Parse the query
            parser = QueryParser("content", schema=self.schema, group=OrGroup)
            query = parser.parse(term)
            results = searcher.search(query, limit=5)

            # for result in results:
            #     print(f"""
            #        Score: {result.score}
            #        Path: {result["path"]}
            #        Doc: {result["content"][:120]}
            #     """)
            return [(result["content"], str(result["path"]), result.score) for result in results]

async def load_docs():
    # opanai = OpenAISearch()
    llama = LlamaSearch()
    whoosh_search = WhooshSearcher()

    for url in [
            "https://go.dev/blog/llmpowered",
            "https://go.dev/blog/alias-names",
            "https://go.dev/blog/gotelemetry",
    ]:
        # openai.load_doc(url)
        await asyncio.gather(
            llama.load_doc(url),
            whoosh_search.load_doc(url)
        )

async def load_all_docs_from_go_blog():
    llama = LlamaSearch()
    whoosh_search = WhooshSearcher()

    urls = fetch_all_article_links("https://go.dev/blog/all")
    semaphore = asyncio.Semaphore(4)
    limited_load_doc = limited_async(llama.load_doc, semaphore)
    tasks = [limited_load_doc(url) for url in urls]
    await asyncio.gather(*tasks)
    print(f"Loaded {len(urls)} into data stores")

# Reusable function for limiting concurrency and handling exceptions
def limited_async(func, semaphore):
    async def wrapper(*args, **kwargs):
        async with semaphore:  # Limit concurrency
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                return None  # Return a default value or re-raise the exception if needed
    return wrapper

def search_and_report(term: str):
    llama = LlamaSearch()
    whoosh_search = WhooshSearcher()

    # print("### llama search report ###")
    # print(f"### Term : {term} ###")
    vector_search_result = llama.search(term)
    print(f"=== Got {len(vector_search_result)} documents from vector search")

    # print("### Whoosearch report ###")
    # print(f"### Term : {term} ###")
    text_search_result = whoosh_search.search(term)
    print(f"=== Got {len(text_search_result)} documents from text search")
    res = construct_results(vector_search_result, text_search_result)
    data = normalize_results(res)
    # plot_histogram(data)
    # plot_scatter_plot(data)
    # plot_heatmap(data)
    correlation = data["Normalized Vector Score"].corr(data["Normalized Text Score"])
    print(f"Correlation between scores: {correlation}")

def search_and_rerank(term: str):
    llama = LlamaSearch()
    whoosh_search = WhooshSearcher()
    vector_search_result = llama.search(term)
    text_search_result = whoosh_search.search(term)
    docs = {}
    for doc in vector_search_result:
        docs[doc[1]] = doc[0]
    for doc in text_search_result:
        docs[doc[1]] = doc[0]
    data = normalize_results(construct_results(vector_search_result, text_search_result))
    ranks = rerank(data)
    res = []
    for rank in ranks:
        res.append((rank[0], docs[rank[0]], rank[1]))
    for item in res:
        print(f"""
            Score: {item[2]}
            Path: {item[0]}
            Doc: {item[1][0:120]}
        """)
    return res

def rerank(data: pd.DataFrame):
    # Step 1: Define prior probability of relevance
    P_relevance = 0.5  # Assume uniform prior

    # Step 2: Fit Gaussian distributions for likelihoods
    # Vector scores
    mu_vector = data["Vector Score"].mean()
    sigma_vector = data["Vector Score"].std()

    # Text scores
    mu_text = data["Text Score"].mean()
    sigma_text = data["Text Score"].std()

    # Step 3: Calculate likelihoods
    data["P(Score1 | Relevance)"] = norm.pdf(data["Vector Score"], loc=mu_vector, scale=sigma_vector)
    data["P(Score2 | Relevance)"] = norm.pdf(data["Text Score"], loc=mu_text, scale=sigma_text)

    # Step 4: Calculate joint likelihood (assuming independence of scores)
    data["Joint Likelihood"] = data["P(Score1 | Relevance)"] * data["P(Score2 | Relevance)"]

    # Step 5: Compute posterior probability (unnormalized)
    data["Posterior"] = data["Joint Likelihood"] * P_relevance

    # Normalize posterior to ensure it sums to 1 (if necessary)
    data["Posterior"] /= data["Posterior"].sum()

    # Step 6: Rank documents based on posterior
    data = data.sort_values(by="Posterior", ascending=False)

    # Display the ranked results
    # print(data[["Path", "Posterior"]])
    return list(zip(data["Path"], data["Posterior"]))

def construct_results(vsr: list[(str,str, float)], tsr:list[(str, str, float)]):
    res = {}
    for doc in vsr:
        path = doc[1]
        res[path] = {"vss": doc[2], "tss": 0}

    for doc in tsr:
        path = doc[1]
        if path in res:
            res[path]["tss"] = doc[2]
        else:
            res[path] = {"tss": doc[2], "vss": 0}
    return res

def normalize_results(inp: dict):
    keys = inp.keys()
    v_ss = []
    t_ss = []
    for key in keys:
        v_ss.append(inp[key]["vss"])
        t_ss.append(inp[key]["tss"])

    data = pd.DataFrame({
        "Path": keys,
        "Vector Score": v_ss,
        "Text Score": t_ss,
    })
    # Min-Max Normalization
    data["Normalized Vector Score"] = (data["Vector Score"] - data["Vector Score"].min()) / (data["Vector Score"].max() - data["Vector Score"].min())
    data["Normalized Text Score"] = (data["Text Score"] - data["Text Score"].min()) / (data["Text Score"].max() - data["Text Score"].min())

    return data

def plot_histogram(data: pd.DataFrame):
    plt.hist(data["Normalized Vector Score"], bins=10, alpha=0.5, label="Vector Scores")
    plt.hist(data["Normalized Text Score"], bins=10, alpha=0.5, label="Text Scores")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Score Distributions")
    plt.show()

def plot_scatter_plot(data: pd.DataFrame):
    plt.scatter(data["Normalized Vector Score"], data["Normalized Text Score"], alpha=0.7)
    plt.xlabel("Normalized Vector Scores")
    plt.ylabel("Normalized Text Scores")
    plt.title("Score Comparison")
    plt.show()

def plot_heatmap(data: pd.DataFrame):
    sns.kdeplot(
    x=data["Normalized Vector Score"],
    y=data["Normalized Text Score"],
    cmap="Blues",
    fill=True
    )
    plt.xlabel("Normalized Vector Scores")
    plt.ylabel("Normalized Text Scores")
    plt.title("Score Density Heatmap")
    plt.show()


async def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Command line tool with load and search functionality.")

    # Define the subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add 'load' subcommand
    subparsers.add_parser("load", help="Triggers the load function")

    # Add load go blog command
    subparsers.add_parser("load_go_blog", help="Triggers the load for all articles in the go blog")

    # Add 'search' subcommand with an argument for the search term
    search_and_report_parser = subparsers.add_parser("search_and_report", help="Triggers the search function with a term")
    search_and_report_parser.add_argument("term", type=str, help="Term to search for")

    search_and_rerank_parser = subparsers.add_parser("search_and_rerank", help="Triggers the search function with a term")
    search_and_rerank_parser.add_argument("term", type=str, help="ranrk to search for")
    # Parse the arguments
    args = parser.parse_args()

    # Handle the commands
    if args.command == "load":
        await load_docs()
    if args.command == "load_go_blog":
        await load_all_docs_from_go_blog()
    elif args.command == "search_and_report":
        search_and_report(args.term)
    elif args.command == "search_and_rerank":
        search_and_rerank(args.term)
    else:
        parser.print_help()

if __name__ == "__main__":
    asyncio.run(main())
