import sys
import bs4
import uuid
import requests
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import HTMLHeaderTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import Optional
from typing import List
import sqlite3
import chromadb

model_name = "llama3.2"
embeddings = OllamaEmbeddings(
    model=model_name,
)

class Queries(BaseModel):
    queries: List[str] = Field(description="Multiple queries derived from the same input query")

def init_sqlitedb():
    # Step 1: Create SQLite Database and Table
    db_path = "documents.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # Create a table to store documents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL
    )
    """)

def get_db():
    db_path = "documents.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    return cursor

def db_insert(doc: str):
    db_path = "documents.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents (content) VALUES (?)", (doc))
    conn.commit()

def init_llm():
    global llm
    global structured_llm
    llm = ChatOllama(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(Queries)

def custom_filter(name, attr):
    # Match tags by name (h1, h2, p) OR by specific classes
    return (
        True
    )

class State(TypedDict):
    question:               str
    queries:                [str]
    docs:                   dict[str, Document]      # Doc ID -> doc
    scores:                 dict[str, float]    # Doc ID -> score
    ranks:                  dict[str, int]      # Doc ID -> rank
    context:                list[Document]
    answer:                 str

def retrieve(state: State):
    scores = {}
    docs = {}
    ranks = {}
    rank = 0
    for query in state["queries"]:
        retrieved_docs_with_scores = vector_store.similarity_search_with_relevance_scores(
                query,
                k=3
            )
        for doc in retrieved_docs_with_scores:
            rank += 1
            doc_id = hash(doc[0].page_content)
            docs[doc_id] = doc[0]
            scores[doc_id] = doc[1]
            ranks[doc_id] = rank

    return {"scores": scores, "docs": docs, "ranks": ranks}

def rerank(state: State):
    context = []
    for doc_id in state["docs"].keys():
        rank = state["ranks"][doc_id]
        score = state["scores"][doc_id]
        updated_score = score + 1 / (rank + 1)
        state["scores"][doc_id] = updated_score

    sorted_scores = {k: v for k, v in sorted(state["scores"].items(), key=lambda item: item[1], reverse=True)}
    i = 0
    for doc_id in sorted_scores.keys():
        i += 1
        state["ranks"][doc_id] = i
        if i < 3:
            context.append(state["docs"][doc_id])
    return {"context": context}


def generate_queries(state: State):
    prompt = """
You are an expert query generator for a retrieval-augmented generation system. Your task is to take a single query and rewrite it into multiple specific and focused queries that capture different aspects or interpretations of the original. Each rewritten query should aim to retrieve complementary information from a large document database.

For example:

    If the original query is broad, generate more specific queries focusing on subtopics.
    If the original query is complex or multi-hop, break it into simpler queries targeting individual pieces of information.
    Ensure the queries are clear, concise, and semantically aligned with the original intent.

Now, please generate {k} alternative queries for the following input query:

Input Query: '{query}'
    """
    prompt = prompt.format(query=state["question"], k=8)
    try:
        response = structured_llm.invoke(prompt)
        if len(response.queries) < 1:
            return {"queries": [query]}
        response.queries.append(state["question"])
        return {"queries": response.queries}
    except:
        return {"queries": [state["question"]]}


def generate(state: State):
    # Prompt
    rag_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the context doesn't provide an answer, just say that you don't know and nothing more and never leak the context or what topics it covers. Don't mention in the answer that you have context.
    
    Question: {question}
    
    Context: {context}
    
    Answer:
    """
    try:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = rag_prompt.format(question = state["question"], context= docs_content)
        response = llm.invoke(messages)
        return {"answer": response.content}
    except:
        return {"answer": "I don't know how to answer this question"}

def init_vs():
    ### Vector store
    persistent_client = chromadb.PersistentClient()
    collection = persistent_client.get_or_create_collection("golang_kb")
    global vector_store
    vector_store = Chroma(
        client=persistent_client,
        collection_name="golang_kb",
        embedding_function=embeddings,
        # Set chroma to use cosine similarity instead of Squared L2
        collection_metadata={"hnsw:space": "cosine"}
    )

def load_doc(url: str):
    init_vs()
    # Split loaded docs
    headers_to_split_on = [
        ("h1", "Header 1"),
        ("h2", "Header 2"),
        ("h3", "Header 3"),
        ("h4", "Header 4"),
    ]

    text_splitter = HTMLHeaderTextSplitter(headers_to_split_on,return_each_element=True)
    all_splits = text_splitter.split_text_from_url(url)
    for split in all_splits:
        print(f"{split}")
    print(f"### Splitted documents into {len(all_splits)} splits")
    uuids = [str(uuid.uuid4()) for _ in range(len(all_splits))]
    vector_store.add_documents(documents=all_splits, ids=uuids)

if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        sys.exit()
    if args[1] == "load":
        url = sys.argv[2]
        init_sqlitedb()
        load_doc(url)
        sys.exit()

    if sys.argv[1] == "start":
        init_vs()
        init_llm()
        # Compile application and test
        graph_builder = StateGraph(State).add_sequence([generate_queries, retrieve, rerank, generate])
        graph_builder.add_edge(START, "generate_queries")
        graph = graph_builder.compile()

        while True:
            user_input = input("> How can I help you? \n ")
            if user_input == "exit()":
                sys.exit()
            response = graph.invoke({"question": user_input})
            print(f"> {response["answer"]}")

    print("Invalid command")
