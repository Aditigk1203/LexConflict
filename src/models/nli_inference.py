from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


class NLIPairDataset(Dataset):

    def __init__(
        self,
        pairs: List[Dict],
        tokenizer,
        max_length: int = 120
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):

        pair = self.pairs[index]

        hypothesis = str(
            pair["hypothesis"]
        )

        evidence = str(
            pair.get(
                "evidence",
                pair.get("clause_text", "")
            )
        )

        encoded = self.tokenizer(
            hypothesis,
            evidence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "pair_index": index
        }


class LegalBERTNLI:

    def __init__(
        self,
        model_path: str,
        device: str = None
    ):

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        print("Loading Legal-BERT model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        )

        self.model.to(self.device)
        self.model.eval()

        print("Using device:", self.device)

        self.id2label = {
            int(k): v
            for k, v in self.model.config.id2label.items()
        }

        print("Label mapping:")
        print(self.id2label)

    def predict(
        self,
        pairs: List[Dict],
        batch_size: int = 32,
        max_length: int = 120,
        num_workers: int = 0
    ):

        dataset = NLIPairDataset(
            pairs,
            self.tokenizer,
            max_length=max_length
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )

        predictions = []

        with torch.no_grad():

            for batch in loader:

                input_ids = batch[
                    "input_ids"
                ].to(self.device)

                attention_mask = batch[
                    "attention_mask"
                ].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probabilities = torch.softmax(
                    outputs.logits,
                    dim=-1
                )

                confidence, predicted_ids = torch.max(
                    probabilities,
                    dim=-1
                )

                for pred_id, conf, probs in zip(
                    predicted_ids.cpu().tolist(),
                    confidence.cpu().tolist(),
                    probabilities.cpu().tolist()
                ):

                    predictions.append({
                        "label_id": int(pred_id),
                        "label": self.id2label[int(pred_id)],
                        "confidence": float(conf),
                        "probabilities": [
                            float(x) for x in probs
                        ]
                    })

        return predictions