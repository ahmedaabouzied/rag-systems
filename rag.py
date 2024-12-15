import sys
import bs4
import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain_chroma import Chroma

model_name = "llama3.2:3b-instruct-fp16"
embeddings = OllamaEmbeddings(
    model=model_name,
)

def init_llm():
    global llm
    llm = ChatOllama(model=model_name, temperature=0)
    llm_json_mode = ChatOllama(model=model_name, temperature=0, format="json")

def custom_filter(name, attr):
    # Match tags by name (h1, h2, p) OR by specific classes
    return (
        "SiteContent SiteContent--default" in attr.get("class", []) or
        "post-content" in attr.get("class", []) or
        "Doc Article" in attr.get("class", []) or
        "post-title" in attr.get("class", []) or
        "post-header" in attr.get("class", [])
    )



# Prompt
rag_prompt = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}

Answer:
"""

class State(TypedDict):
    question: str
    context: list[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"], k=8)
    print(f"### Retrieved {len(retrieved_docs)} docs related to the question")
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format(question = state["question"], context= docs_content)
    response = llm.invoke(messages)
    return {"answer": response.content}

def init_vs():
    ### Vector store
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection("golang_kb")
    global vector_store
    vector_store = Chroma(
        client=persistent_client,
        collection_name="golang_kb",
        embedding_function=embeddings,
    )

def load_doc(url: str):
    init_vs()
    # Loading data from HTML resource over HTTP
    loader = WebBaseLoader(
        web_paths=(
            url,
        ),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                custom_filter
            )
        ),
    )
    docs = loader.load()

    print(f"### Loaded {len(docs)} web documents")
    # Split loaded docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    print(f"### Splitted documents into {len(all_splits)} splits")
    vector_store.add_documents(documents=all_splits)

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        sys.exit()
    if args[1] == "load":
        url = sys.argv[2]
        load_doc(url)
        sys.exit()

    if sys.argv[1] == "start":
        init_vs()
        init_llm()
        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        while True:
            user_input = input("How can I help you \n")
            if user_input == "exit()":
                sys.exit()
            response = graph.invoke({"question": user_input})
            print(response["answer"])

    print("Invalid command")
