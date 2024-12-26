from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel, PreTrainedTokenizer
import torch

class Checker():
    labels: dict[int, str]
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizer

    def __init__(self, model_name="microsoft/deberta-large-mnli"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = {0: "contradiction", 1: "neutral", 2:"entailment"}

    def check_contradiction(self, premise: str, hypothesis: str):
        # Tokenize the inputs
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt")

        # Get model output
        outputs = self.model(**inputs)
        logits = outputs.logits

        # Apply softmax to the logits to get probabilies
        probabilities = torch.softmax(logits, dim=-1)

        # Get the index of the max probability
        predicted_index = int(torch.argmax(probabilities).item())
        predicted_label = self.labels[predicted_index]

        # Get the value of the max probability
        confidence = probabilities.max().item()

        return predicted_label, confidence

def test_check_contradiction_entailment():
    checker = Checker()
    res = checker.check_contradiction("Go supports generics", "Go has generics")
    assert res[0] == "entailment"
    assert res[1] > 0.9
