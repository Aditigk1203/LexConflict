import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class LegalBERTInference:

    def __init__(
        self,
        model_path,
        max_length=512
    ):

        self.model_path = model_path
        self.max_length = max_length

        print("Loading Legal-BERT...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model.to(self.device)
        self.model.eval()

        print(
            f"Legal-BERT loaded on {self.device}"
        )

        print(
            "Label mapping:"
        )

        print({
            0: "Entailment",
            1: "NotMentioned",
            2: "Contradiction"
        })


    def predict(
        self,
        clause_a,
        clause_b
    ):

        encoded = self.tokenizer(
            clause_a,
            clause_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        with torch.no_grad():

            output = self.model(
                **encoded
            )

            probabilities = torch.softmax(
                output.logits,
                dim=-1
            )[0]

        label_id = int(
            torch.argmax(probabilities).item()
        )

        label_map = {
            0: "Entailment",
            1: "NotMentioned",
            2: "Contradiction"
        }

        return {
            "label_id": label_id,

            "label":
                label_map[label_id],

            "confidence":
                float(probabilities[label_id].item()),

            "probabilities": [
                float(x)
                for x in probabilities.cpu().tolist()
            ]
        }