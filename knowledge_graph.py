import ast
import langchain_ollama
class RelationExtractor:
    model_name: str
    llm: langchain_ollama.OllamaLLM
    re_prompt = """ 
        You're an assistant for extracting entities and relations from given paragraphs. Follow the examples below to return the relations. Dont's how me the steps. Only return the results.

        Example:
        Input: "Go supports generics."
        Output: [("Go", "support", "generics")]

        Example:
        Input: "Robert likes chocolate."
        Output: [("Robert", "like" "chocolate")]

        Example:
        Input: "Goroutines communicate with each other through channels in Go."
        Output: [
                ("Goroutines", "communicate through", "channels"),
                ("Goroutiens", "communicate with", "each other")
                ("Gorouties, "in", "Go"),
                ("Channels", "in", "Go"),
            ]

        Example:
        Input: "The net/http package allows building HTTP servers and clients in Go applications."
        Output: [
                ("net/http package", "allows building", "HTTP server")
                ("net/http package", "allows building", "HTTP client")
            ]
        Example:
        Input: "Go modules simplify dependency management in Go projects."
        Output: [
            ("Go modules", "simplify", "dependency management"),
            ("modules", "in", "Go"),
            ("dependency management", "in", "Go projects"),
        ]

        Task:
        Input: {input}
        Output:
    """

    def __init__(self, model_name="llama3.2"):
        self.model_name = model_name
        self.llm = langchain_ollama.OllamaLLM(model=model_name)

    def _remove_filler_words(self, sentence: str):
        sentence.replace("The", "")
        sentence.replace("the", "")
        sentence.replace(" a ", "")
        return sentence

    def _parse_result(self, res: str):
        try:
            relations: list[str] = ast.literal_eval(res.strip())
            return relations
        except Exception as e:
            raise Exception(f"failed to parse into tuple list: {e}")

    def extract_relations(self, sentence: str):
        try:
            sentence = self._remove_filler_words(sentence)
            prompt = self.re_prompt.format(input=sentence)
            return self._parse_result(self.llm.invoke(prompt))
        except Exception as e:
            print(f"error extracting relations: {e}")
            return None

def main():
    re = RelationExtractor("llama3.2")
    for sentence in [
            "Go supports iterators.",
            "The sync package provides synchronization primitives for concurrent programming.",
            "Go modules simplify dependency management in Go projects.",
            "The io.Writer interface enables writing data to various destinations.",
            "Java supports multithreading to allow concurrent execution of tasks.",
            "The Spring framework simplifies dependency injection in Java applications.",
    ]:
        relations = re.extract_relations(sentence)
        if relations:
            for relation in relations:
                print(relation)

if __name__ == "__main__":
    main()
