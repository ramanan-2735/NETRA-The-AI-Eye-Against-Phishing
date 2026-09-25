# -*- coding: utf-8 -*-
"""
NETRA - Tier-2 RoBERTa Model Module
=====================================
Loads the fine-tuned RoBERTa-base model for 3-class phishing escalation.
Used by the Tier-2 FastAPI service.
"""

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import RobertaModel, AutoTokenizer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model architecture (must match train_roberta_tier2.py)
# ---------------------------------------------------------------------------
class RoBERTaTier2Classifier(nn.Module):
    def __init__(self, num_classes=3, dropout=0.3):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        self.header_projection = nn.Linear(10, 32)
        # 768 (CLS) + 32 (header) + 1 (tier1_confidence) = 801
        self.classifier = nn.Sequential(
            nn.Linear(801, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask, header_features, tier1_confidence):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        header_proj = torch.relu(self.header_projection(header_features))  # (batch, 32)
        tier1_conf = tier1_confidence.unsqueeze(1) if tier1_confidence.dim() == 1 else tier1_confidence
        combined = torch.cat([cls_output, header_proj, tier1_conf], dim=1)  # (batch, 801)
        return self.classifier(combined)  # (batch, num_classes)


# ---------------------------------------------------------------------------
# Model manager
# ---------------------------------------------------------------------------
class Tier2Model:
    CLASS_NAMES = ["LEGITIMATE", "SUSPICIOUS", "PHISHING"]

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self.config = None
        self.ready = False

    def load(self):
        config_path = self.model_dir / "roberta_tier2_config.json"
        model_path  = self.model_dir / "roberta_tier2.pt"
        tok_path    = self.model_dir / "roberta_tier2_tokenizer"

        if not model_path.exists():
            log.info("roberta_tier2.pt not found locally. Auto-downloading from GitHub Release (v3.0.0)...")
            try:
                import urllib.request
                RELEASE_URL = "https://github.com/ramanan-2735/NETRA-The-AI-Eye-Against-Phishing/releases/download/v3.0.0/roberta_tier2.pt"
                urllib.request.urlretrieve(RELEASE_URL, str(model_path))
                log.info("roberta_tier2.pt downloaded successfully from GitHub Release!")
            except Exception as e:
                log.error(f"Failed to auto-download roberta_tier2.pt from GitHub: {e}")
                return False

        log.info("Loading Tier-2 RoBERTa model...")

        with open(config_path) as f:
            self.config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(str(tok_path))

        num_classes = self.config.get("num_classes", 3)
        self.model = RoBERTaTier2Classifier(num_classes=num_classes)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        self.ready = True

        log.info(f"Tier-2 model loaded on {self.device}")
        return True

    def predict(self, text: str, header_features: list, tier1_confidence: float):
        if not self.ready:
            raise RuntimeError("Model not loaded")

        max_len = self.config.get("max_length", 512)

        enc = self.tokenizer(
            text, max_length=max_len, truncation=True,
            padding="max_length", return_tensors="pt",
        )

        hdr_tensor = torch.tensor([header_features], dtype=torch.float32)
        conf_tensor = torch.tensor([tier1_confidence], dtype=torch.float32)

        with torch.no_grad():
            logits = self.model(
                enc["input_ids"].to(self.device),
                enc["attention_mask"].to(self.device),
                hdr_tensor.to(self.device),
                conf_tensor.to(self.device),
            )
            proba = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_class = int(proba.argmax())
        confidence = float(proba.max())
        risk_score = float(proba[2])  # phishing probability

        return {
            "verdict":    self.CLASS_NAMES[pred_class],
            "risk_score": round(risk_score, 4),
            "confidence": round(confidence, 4),
            "class_probabilities": {
                name: round(float(p), 4)
                for name, p in zip(self.CLASS_NAMES, proba)
            },
        }
