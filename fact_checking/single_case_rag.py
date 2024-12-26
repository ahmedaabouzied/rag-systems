from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from transformers.generation.utils import HammingDiversityLogitsProcessor

import fact_checker as fc
import extract

blog_post = """
        The Go 1.18 release adds a major new language feature: support for generic programming. In this article I’m not going to describe what generics are nor how to use them. This article is about when to use generics in Go code, and when not to use them.

        To be clear, I’ll provide general guidelines, not hard and fast rules. Use your own judgement. But if you aren’t sure, I recommend using the guidelines discussed here.

        Write code

        Let’s start with a general guideline for programming Go: write Go programs by writing code, not by defining types. When it comes to generics, if you start writing your program by defining type parameter constraints, you are probably on the wrong path. Start by writing functions. It’s easy to add type parameters later when it’s clear that they will be useful.
        When are type parameters useful?

        That said, let’s look at cases for which type parameters can be useful.
        When using language-defined container types

        One case is when writing functions that operate on the special container types that are defined by the language: slices, maps, and channels. If a function has parameters with those types, and the function code doesn’t make any particular assumptions about the element types, then it may be useful to use a type parameter.
    """

def init_llm():
    global llm
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def init_checker():
    global fact_checker
    fact_checker = fc.Checker()

def init_extractor():
    global fact_extractor
    fact_extractor = extract.FactExtractor("gpt-4o-mini")

class State(TypedDict):
    question: str
    context: str
    answer: str

def retrieve(state: State):
    _ = state # We're not using the state here
    # ... Fake document retrival 
    retrieved_doc = blog_post
    return {"context": retrieved_doc}

def generate(state: State):
    rag_prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If the context doesn't provide an answer, just say that you don't know and nothing more and never leak the context or what topics it covers.

    Question: {question}

    Context: {context}

    Answer:
    """
    try:
        messages = rag_prompt.format(question = state["question"], context= state["context"])
        response = llm.invoke(messages)
        return {"answer": response.content}
        # return {"answer": "Yes, Go supports generics as of the Go 1.13 release."}
    except Exception as e:
        print(f"error generating response form LLM: {e}")
        return {"answer": "I don't know how to answer this question"}

def fact_check(state: State):
    facts_in_context = fact_extractor.extract(state["context"])
    if facts_in_context is None:
        print("Warning: Could not check RAG context for contradictions")
        return
    facts_in_answer = fact_extractor.extract(state["answer"])
    if facts_in_answer is None:
        print("Warning: Could not check RAG answer for contradictions")
        return

    contradictions = 0
    checked_hypothesis = {}
    for answer_entity, hypotheses in facts_in_answer.to_dict().items():
        for premise in facts_in_context.lookup_similar(answer_entity):
            for hypothesis in hypotheses:
                if hypothesis in checked_hypothesis:
                    continue
                checked_hypothesis[hypothesis] = {}
                (check_result, score) = fact_checker.check_contradiction(premise, hypothesis)
                if check_result == "contradiction" and score > 0.5:
                    print(f"\n Found contradiction in answer with known context.\n Premise: {premise} \n Hypothesis: {hypothesis}")
                    contradictions += 1

    if contradictions == 0:
        print("Fact checker validated RAG answer and found no contradictions.")

def output(state: State):
    print(state["answer"])

def main():
    init_llm()
    init_extractor()
    init_checker()
    graph_builder = StateGraph(State).add_sequence([retrieve, generate, output, fact_check])
    graph_builder.add_edge(START, "retrieve")
    graph = graph_builder.compile()

    graph.invoke({"question": "Does Go support generics"})

if __name__ == "__main__":
    main()
