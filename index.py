from whoosh.index import open_dir
from whoosh.qparser import QueryParser

# Open the index
index = open_dir("woosh_index")

# Query the index
with index.searcher() as searcher:
    # Parse the query
    query = QueryParser("content", index.schema).parse("alias declaration")
    
    # Search the index
    results = searcher.search(query, limit=None)  # Retrieve all results
    
    # Iterate through the results and print scores
    for hit in results:
        print(f"Document ID: {hit.docnum}, Score: {hit.score}, Content: {hit['content'][0:100]}")

