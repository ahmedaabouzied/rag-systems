from typing import Generic, TypeVar
import langchain_ollama
import langchain_openai
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field
import getpass
import os
from abc import ABC, abstractmethod


LLAMA_MODEL     = "llama3.2"
OPEN_AI_MODEL   = "gpt-4o-mini"

class Extract(BaseModel):
    """ Extract is a list of entities, concepts, and facts from a given text """
    entities: list[str] = Field(description="Entities can be names, concepts, terms, objects, ...")
    concepts: list[str] = Field(description="General concepts that are not named entities")
    facts: list[str] = Field(description="Facts related to the entities or the concepts")

class Entity(BaseModel):
    entity: str = Field(description="An entity")
    sentences: list[str] = Field(description="Sentences that mention the entity")

class Facts(BaseModel):
    entities: list[Entity] = Field()

    def to_dict(self) -> dict[str, list[str]]:
        res = {}
        for entity in self.entities:
            res[entity.entity] = entity.sentences
        return res

    def lookup_similar(self, key: str) -> list[str]:
        index = self.to_dict()
        res: list[str] = [];
        parts = key.split(" ")
        parts += key.split("-")
        parts += key.split("_")
        parts += key.split(",")
        for part in parts:
            for key in index.keys():
                if part in key:
                    res += index[key]
        return res

class Relation(BaseModel):
    head: str = Field(description="Entity one of the relation. Head in a graph representation")
    tail: str = Field(description="Entity two of the relation. Tail in a graph representation")
    edge: str = Field(description="The description of the relation. Edge between two nodes in a graph representation")

    def __str__(self):
        return f"({self.head} , {self.edge}, {self.tail}) \n"

    def __repr__(self):
        return f"({self.head} , {self.edge}, {self.tail}) \n"

class Relations(BaseModel):
    relations: list[Relation] = Field()

    def __str__(self):
        return f"{[f"{relation} \n" for relation in self.relations]}"

    def __repr__(self):
        return f"{[f"{relation} \n" for relation in self.relations]}"

T = TypeVar('T', bound=BaseModel) # Generic <T>

class _Extractor(ABC, Generic[T]):
    """ Extractor is a base class to extract info from a text using an LLM """
    # _model_name
    def set_model_name(self, model_name:str = "ollama3.2"):
        self._model_name = model_name

    def get_model_name(self):
        return self._model_name

    # _llm
    def set_llm(self, llm: Runnable):
        self._llm = llm

    def get_llm(self):
        return self._llm

    # _base_prompt
    def set_base_prompt(self, base_prompt: str):
        self._base_prompt = base_prompt

    def get_base_prompt(self):
        return self._base_prompt

    @abstractmethod
    def extract(self, paragraph: str) -> T | None:
        pass

extract_entities_prompt = """
        Role: You're an assistant for extracting all possible entities, all possible concepts, and all possible facts from a paragraph. Facts should be full sentences taken from the input paragraph. Don't show the stpes. Only return the results.

        Rules:
            When you see words like "This" replace them with what they refer to
        Input: {input}
    """

extract_relations_prompt = """
    Roles: You're an assistant for extracting entities and relations from given paragraphs.
    Represent the relation as 2 knowledge graph nodes connected by the sentence as the edge.
    For each sentence in the paragraph, reduce the complexity of the sentence by turning it into smaller sentences, for each smaller sentence, extract the 2 entities that represent the subject and object and the sentence should be the relation between them.
    The output should be (<node 1> , relation , <node 2>

    How to do it?
        - Reduce the lexical complexity of sentences by turning a large sentence into
            smaller sentences with only subject, verb, and object.
        - Represent the smaller sentences as relation between 2 entities.
        - When you see words like "This", replace them with what they refer to.
        - The relation should be meaningful so that it gets inserted to a knowledge graph.
        - The entities should be items, terms, or concepts.
    Example:
        Input: "Go supports generics."
        Output: [("Go", "Go supports generics", "generics")]

    Example:
        Input: "Goroutines communicate with each other through channels in Go."
        Output: [
            ("Goroutines", "Goroutines communicate through channels", "channels"),
            ("Goroutiens", "Goroutines communicate with each other", "each other")
            ("Gorouties, "Gouroutines is a feature in Go", "Go"),
            ("Channels", "Channels is a feature in Go", "Go"),
        ]
    Example:
        Input: "Smaller, more manageable pieces of code are connected through public APIs. In Go, these APIs consist of the identifiers exported by a package."
        Output: [
            ("Smaller, more managable pieces of code", "Smaller, more managable pieces of code are connected through public APIs", "Public APIs"),
            "("pieces of code", "Pieces of code are connected through public APIs", "Public APIs")
            "("Public APIs", "Consist of identifiers exported by a package", "identifiers")
            "("Identifiers", "Identifiers are exported by a package", "package")
        ]

    Input: {input}
"""

extract_facts_prompt = """
    Role: You'e an assistant for extracting entities and facts about thos intenties from a piece of text.
    Given a paragraph, extract all meaningful entities in it and the facts mentioned in the text about those entities.

    Input: {input}
"""


class FactExtractor(_Extractor[Facts]):
    def __init__(self, model_name="llama3.2"):
        self.set_model_name(model_name)
        self.set_base_prompt(extract_facts_prompt)
        if model_name == LLAMA_MODEL:
            self.set_llm(langchain_ollama.ChatOllama(model=model_name).with_structured_output(Facts))
        elif model_name == OPEN_AI_MODEL:
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
            self.set_llm(langchain_openai.ChatOpenAI(model=model_name).with_structured_output(Facts))

    def extract(self, paragraph: str) -> Facts | None:
        """ Extracts the information from a given text and formats the output based on the return type """
        try:
            prompt = self.get_base_prompt().format(input=paragraph)
            return self.get_llm().invoke(prompt)
        except Exception as e:
            print(f"error extracting facts: {e}")
            return None


class RelationExtractor(_Extractor[Relations]):
    def __init__(self, model_name="llama3.2"):
        self.set_model_name(model_name)
        self.set_base_prompt(extract_facts_prompt)
        if model_name == LLAMA_MODEL:
            self.set_llm(langchain_ollama.ChatOllama(model=model_name).with_structured_output(Relations))
        elif model_name == OPEN_AI_MODEL:
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
            self.set_llm(langchain_openai.ChatOpenAI(model=model_name).with_structured_output(Relations))

    def extract(self, paragraph: str) -> Relations | None:
        """ Extracts the information from a given text and formats the output based on the return type """
        try:
            prompt = self.get_base_prompt().format(input=paragraph)
            return self.get_llm().invoke(prompt)
        except Exception as e:
            print(f"error extracting relations: {e}")
            return None

class EntitiesExtractor(_Extractor[Entity]):
    def __init__(self, model_name="llama3.2"):
        self.set_model_name(model_name)
        self.set_base_prompt(extract_facts_prompt)
        if model_name == LLAMA_MODEL:
            self.set_llm(langchain_ollama.ChatOllama(model=model_name).with_structured_output(Entity))
        elif model_name == OPEN_AI_MODEL:
            if not os.environ.get("OPENAI_API_KEY"):
                os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")
            self.set_llm(langchain_openai.ChatOpenAI(model=model_name).with_structured_output(Entity))

    def extract(self, paragraph: str) -> Entity | None:
        """ Extracts the information from a given text and formats the output based on the return type """
        try:
            prompt = self.get_base_prompt().format(input=paragraph)
            return self.get_llm().invoke(prompt)
        except Exception as e:
            print(f"error extracting relations: {e}")
            return None


def test_extractor_extract_relations_with_open_ai():
    re = RelationExtractor(OPEN_AI_MODEL)
    for paragraph in [
            "Go supports iterators.",
            "The sync package provides synchronization primitives for concurrent programming.",
            "Go modules simplify dependency management in Go projects.",
            "The io.Writer interface enables writing data to various destinations.",
            "Java supports multithreading to allow concurrent execution of tasks.",
            "The Spring framework simplifies dependency injection in Java applications.",
            """Go was designed for programming at scale. Programming at scale means dealing with large amounts of data, but also large codebases, with many engineers working on those codebases over long periods of time. Go’s organization of code into packages enables programming at scale by splitting up large codebases into smaller, more manageable pieces, often written by different people, and connected through public APIs. In Go, these APIs consist of the identifiers exported by a package: the exported constants, types, variables, and functions. This includes the exported fields of structs and methods of types.""",
    ]:
        attempt = 1
        out = re.extract(paragraph)
        while out is None and attempt < 3:
            attempt += 1
            out = re.extract(paragraph)
        if out is None:
            raise Exception("Retried LLM query for {attempt} times and got None")

        print(out.relations)

def test_extractor_extract_facts_with_open_ai():
    re = FactExtractor(OPEN_AI_MODEL)
    for paragraph in [
            "Go supports iterators.",
            "The sync package provides synchronization primitives for concurrent programming.",
            "Go modules simplify dependency management in Go projects.",
            "The io.Writer interface enables writing data to various destinations.",
            "Java supports multithreading to allow concurrent execution of tasks.",
            "The Spring framework simplifies dependency injection in Java applications.",
            """Go was designed for programming at scale. Programming at scale means dealing with large amounts of data, but also large codebases, with many engineers working on those codebases over long periods of time. Go’s organization of code into packages enables programming at scale by splitting up large codebases into smaller, more manageable pieces, often written by different people, and connected through public APIs. In Go, these APIs consist of the identifiers exported by a package: the exported constants, types, variables, and functions. This includes the exported fields of structs and methods of types.""",
    ]:
        attempt = 1
        out = re.extract(paragraph)
        while out is None and attempt < 3:
            attempt += 1
            out = re.extract(paragraph)
        if out is None:
            raise Exception("Retried LLM query for {attempt} times and got None")

        print(out)
