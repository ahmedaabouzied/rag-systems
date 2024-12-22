import os
import sys
import uuid
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from typing import List
import sqlite3
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import chromadb
import scrap

model_name = "llama3.2"
embeddings = OllamaEmbeddings(
    model=model_name,
)

import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

openAI_embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

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

def db_insert(doc: Document):
    db_path = "documents.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO documents(content) VALUES (?)", (doc.page_content,))
    conn.commit()

def index_docs(cursor):
    # Init woosh index directory
    index_dir = "woosh_index"
    if not os.path.exists(index_dir):
        os.mkdir(index_dir)

    schema = Schema(id=ID(stored=True), content=TEXT(stored=True))

    # Create or open the woosh index
    if not os.listdir(index_dir):
        index = create_in(index_dir, schema)
        print("created index")
    else:
        index = open_dir(index_dir)

    writer = index.writer()
    cursor.execute("SELECT id, content FROM documents")
    for doc_id, content in cursor.fetchall():
        print(f"from sqlite {content}")
        writer.add_document(id=str(doc_id), content=content)

    writer.commit()

def search_index(term: str):
    print(f"Text search over {term}")
    res = []
    index_dir = "woosh_index"
    index = open_dir(index_dir)
    with index.searcher() as searcher:
        query = QueryParser("content", index.schema).parse(term)
        results = searcher.search(query)
        print(f"Text search found {len(results)} docs")
        for result in results:
            res.append((result["content"], result["score"]))
    return res


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
    query = state["question"]
        # Search in vector DB
    retrieved_docs_with_scores = vector_store.similarity_search_with_score(
        query,
        k=4
    )

    for doc in retrieved_docs_with_scores:
        rank += 1
        doc_id = hash(doc[0].page_content)
        docs[doc_id] = doc[0]
        scores[doc_id] = doc[1]
        ranks[doc_id] = rank

    # Search in woosh index
    index_search_results = search_index(query)
    for doc in index_search_results:
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
        # state["scores"][doc_id] = updated_score

    sorted_scores = {k: v for k, v in sorted(state["scores"].items(), key=lambda item: item[1], reverse=True)}
    i = 0
    for doc_id in sorted_scores.keys():
        i += 1
        state["ranks"][doc_id] = i
        context.append(state["docs"][doc_id])
    return {"context": context}


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
    global vector_store
    vector_store = Chroma(
        client=persistent_client,
        collection_name="open_ai_embeddings_golang_posts",
        embedding_function=openAI_embeddings,
        # Set chroma to use cosine similarity instead of Squared L2
        collection_metadata={"hnsw:space": "cosine"}
    )

def load_doc(url: str):
    init_vs()

    _, doc = scrap.fetch_and_clean_html(url)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False
    )
    all_splits = text_splitter.create_documents([doc])
    for split in all_splits:
        db_insert(split)
    index_docs(get_db())
    uuids = [str(uuid.uuid4()) for _ in range(len(all_splits))]
    vector_store.add_documents(documents=all_splits, ids=uuids)
    print(f"Indexed {len(all_splits)} docs into Woosh index")
    print(f"Inserted {len(all_splits)} docs into chromad DB vector store")

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
        graph_builder = StateGraph(State).add_sequence([retrieve, rerank, generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()

        while True:
            user_input = input("> How can I help you? \n ")
            if user_input == "exit()":
                sys.exit()
            response = graph.invoke({"question": user_input})
            print(f"> {response["answer"]}")

    print("Invalid command")
