# -*- coding: utf-8 -*-
"""
NETRA - FastAPI Inference Server v2
=====================================
Serves Tier-1 phishing detection model over a local REST API.
Supports both DistilBERT (Phase 2) and calibrated Random Forest (Phase 1) inference.

Start server:
    uvicorn api.main:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health                 -> server + model status + active thresholds
    POST /predict                -> classification + risk_score + signals
    POST /admin/recalibrate      -> live threshold tuning without restart
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Project root on sys.path (needed for ml.features imports in subprocess)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR             = ROOT / "ml" / "models"
MODEL_CALIBRATED_PATH  = MODELS_DIR / "tier1_model_calibrated.pkl"
MODEL_PATH             = MODELS_DIR / "tier1_model.pkl"
TFIDF_PATH             = MODELS_DIR / "tfidf_vectorizer.pkl"
THRESHOLD_CONFIG_PATH  = MODELS_DIR / "threshold_config.json"
DISTILBERT_CONFIG_PATH = MODELS_DIR / "distilbert_config.json"
DISTILBERT_MODEL_PATH  = MODELS_DIR / "distilbert_tier1.pt"
DISTILBERT_TOKENIZER_PATH = MODELS_DIR / "distilbert_tokenizer"
TIER2_CONFIG_PATH = MODELS_DIR / "tier2_config.json"

# ---------------------------------------------------------------------------
# Safe defaults (matches manual_empirical_v2 patch)
# ---------------------------------------------------------------------------
DEFAULT_PHISHING_THRESHOLD = 0.2500
DEFAULT_SUSPICIOUS_LOWER   = 0.0800

# ---------------------------------------------------------------------------
# Urgency keywords for signal extraction
# ---------------------------------------------------------------------------
URGENCY_KEYWORDS = [
    "urgent", "immediately", "suspended", "verify", "click here",
    "confirm your", "account closure", "within 24 hours", "within 30 minutes",
    "permanent", "act now", "expires today", "validate", "update your",
]

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NETRA — AI-Powered Phishing Detection Engine",
    description="""
### State-of-the-Art DistilBERT Deep Learning Email Security

NETRA Tier-1 Phishing Detection Pipeline:
- **Core Architecture**: Transformer-based Fine-Tuned DistilBERT (66M Parameters)
- **Signal Fusion**: Body text semantics + 10 RFC Header Signals (SPF / DKIM / DMARC) + URL Heuristics
- **Typosquatting Engine**: Inline Levenshtein distance against 20+ top targeted enterprise brands
- **Latency**: Sub-200ms real-time inference
    """,
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000", "http://127.0.0.1",
                   "chrome-extension://*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# Rate Limiting (20 requests per minute per client IP)
# ---------------------------------------------------------------------------
from collections import defaultdict

_rate_limits = defaultdict(list)
RATE_LIMIT_PER_MINUTE = 20
RATE_LIMIT_WINDOW_SECONDS = 60

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Apply rate limiting to inference endpoint (/predict)
    if request.url.path in ["/predict"]:
        # Cloud Run / Proxy support: inspect X-Forwarded-For
        forwarded_for = request.headers.get("x-forwarded-for", "")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        else:
            client_ip = request.client.host if request.client else "unknown"

        now = time.time()
        window_start = now - RATE_LIMIT_WINDOW_SECONDS

        # Prune older timestamps
        timestamps = [t for t in _rate_limits[client_ip] if t > window_start]
        if len(timestamps) >= RATE_LIMIT_PER_MINUTE:
            retry_after = int(RATE_LIMIT_WINDOW_SECONDS - (now - timestamps[0])) + 1
            log.warning(f"Rate limit exceeded for IP: {client_ip} ({len(timestamps)} reqs in window)")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Rate limit exceeded. Maximum {RATE_LIMIT_PER_MINUTE} requests per minute allowed.",
                    "retry_after_seconds": max(retry_after, 1),
                },
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        timestamps.append(now)
        _rate_limits[client_ip] = timestamps

    response = await call_next(request)
    return response

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
class ModelState:
    # RF state
    rf_model         = None
    text_extractor   = None
    rf_ready: bool   = False
    rf_model_type: str = "uncalibrated_rf"

    # DistilBERT state
    db_model         = None
    db_tokenizer     = None
    db_config: dict  = {}
    db_ready: bool   = False

    # Tier-2 state
    tier2_client = None
    tier2_enabled: bool = False
    tier2_config: dict  = {}

    # Active thresholds
    phishing_threshold: float  = DEFAULT_PHISHING_THRESHOLD
    suspicious_lower: float    = DEFAULT_SUSPICIOUS_LOWER
    suspicious_upper: float    = DEFAULT_PHISHING_THRESHOLD
    threshold_source: str      = "default_fallback"

    @property
    def model_type(self) -> str:
        if self.db_ready:
            return "distilbert"
        return self.rf_model_type

    @property
    def model_loaded(self) -> bool:
        return self.db_ready or self.rf_ready

state = ModelState()


# ---------------------------------------------------------------------------
# Threshold config helpers
# ---------------------------------------------------------------------------
def _load_thresholds():
    """Load threshold config from JSON file into state."""
    if THRESHOLD_CONFIG_PATH.exists():
        try:
            with open(THRESHOLD_CONFIG_PATH) as f:
                cfg = json.load(f)
            state.phishing_threshold = cfg.get("phishing_threshold", DEFAULT_PHISHING_THRESHOLD)
            state.suspicious_lower   = cfg.get("suspicious_lower",   DEFAULT_SUSPICIOUS_LOWER)
            state.suspicious_upper   = cfg.get("suspicious_upper",   state.phishing_threshold)
            state.threshold_source   = cfg.get("method", "threshold_config.json")
            log.info(f"Thresholds loaded: phishing>={state.phishing_threshold:.4f} "
                     f"| suspicious [{state.suspicious_lower:.4f}, {state.suspicious_upper:.4f})")
        except Exception as e:
            log.warning(f"Could not read threshold_config.json: {e} — using defaults")
    else:
        log.warning(f"threshold_config.json not found — using defaults: "
                    f"phishing>={DEFAULT_PHISHING_THRESHOLD}, suspicious>={DEFAULT_SUSPICIOUS_LOWER}")


def _save_thresholds():
    """Write current in-memory thresholds back to JSON file."""
    cfg = {
        "phishing_threshold": state.phishing_threshold,
        "suspicious_lower":   state.suspicious_lower,
        "suspicious_upper":   state.suspicious_upper,
        "method":             state.threshold_source,
    }
    with open(THRESHOLD_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


# ---------------------------------------------------------------------------
# Startup: load all artifacts
# ---------------------------------------------------------------------------
@app.on_event("startup")
def load_models():
    log.info("=" * 60)
    log.info("NETRA API v2 starting up...")
    log.info("=" * 60)

    _load_thresholds()

    # Auto-download trained weights if missing (e.g. for teammates cloning fresh repo)
    if not DISTILBERT_MODEL_PATH.exists():
        log.info("distilbert_tier1.pt not found locally. Downloading from official GitHub Release (v2.0.0)...")
        try:
            import urllib.request
            RELEASE_URL = "https://github.com/ramanan-2735/NETRA-The-AI-Eye-Against-Phishing/releases/download/v2.0.0/distilbert_tier1.pt"
            urllib.request.urlretrieve(RELEASE_URL, str(DISTILBERT_MODEL_PATH))
            log.info("distilbert_tier1.pt downloaded successfully from GitHub Release!")
        except Exception as e:
            log.warning(f"Failed to auto-download model from GitHub: {e}")

    # --- Try DistilBERT first ---
    if DISTILBERT_CONFIG_PATH.exists() and DISTILBERT_MODEL_PATH.exists() and DISTILBERT_TOKENIZER_PATH.exists():
        try:
            import torch
            from transformers import AutoTokenizer, DistilBertModel
            import torch.nn as nn

            with open(DISTILBERT_CONFIG_PATH) as f:
                state.db_config = json.load(f)

            class PhishingClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
                    self.header_proj = nn.Linear(10, 32)
                    self.classifier = nn.Sequential(
                        nn.Linear(768 + 32, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 2)
                    )
                def forward(self, input_ids, attention_mask, header_features):
                    cls = self.distilbert(input_ids, attention_mask).last_hidden_state[:, 0, :]
                    hdr = torch.relu(self.header_proj(header_features))
                    return self.classifier(torch.cat([cls, hdr], dim=1))

            db_model = PhishingClassifier()
            db_model.load_state_dict(torch.load(DISTILBERT_MODEL_PATH, map_location="cpu"))
            db_model.eval()

            state.db_tokenizer = AutoTokenizer.from_pretrained(str(DISTILBERT_TOKENIZER_PATH))
            state.db_model     = db_model
            state.db_ready     = True
            log.info("DistilBERT model loaded successfully.")
        except Exception as e:
            log.warning(f"DistilBERT load failed: {e} — falling back to RF")

    # --- Load RF (always load as fallback) ---
    rf_path = MODEL_CALIBRATED_PATH if MODEL_CALIBRATED_PATH.exists() else MODEL_PATH
    if rf_path.exists() and TFIDF_PATH.exists():
        try:
            state.rf_model       = joblib.load(rf_path)
            state.text_extractor = joblib.load(TFIDF_PATH)
            state.rf_ready       = True
            state.rf_model_type  = "calibrated_rf" if MODEL_CALIBRATED_PATH.exists() else "uncalibrated_rf"
            log.info(f"RF model loaded [{state.rf_model_type}]: {rf_path.name}")
        except Exception as e:
            log.error(f"RF model load failed: {e}")

    
    # --- Load Tier-2 config (if available) ---
    if TIER2_CONFIG_PATH.exists():
        try:
            with open(TIER2_CONFIG_PATH) as f:
                state.tier2_config = json.load(f)
            import os
            tier2_url = os.getenv("NETRA_TIER2_URL", "")
            if tier2_url:
                from api.tier2_client import Tier2Client
                state.tier2_client = Tier2Client(base_url=tier2_url)
                state.tier2_enabled = True
                log.info(f"Tier-2 escalation enabled: {tier2_url}")
            else:
                log.info("Tier-2 config found but NETRA_TIER2_URL not set - escalation disabled")
        except Exception as e:
            log.warning(f"Tier-2 config load failed: {e}")

    log.info(f"Active model: {state.model_type} | Loaded: {state.model_loaded}")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    subject: Optional[str] = None
    body_text: str = Field(..., description="Full email body text")
    urls: Optional[List[str]] = Field(default=[], description="URLs extracted from email")
    sender: Optional[str]     = Field(default=None)
    reply_to: Optional[str]   = Field(default=None)
    headers: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Parsed email auth headers. Expected keys: "
            "'spf' ('pass'|'fail'|'none'|'neutral'), "
            "'dkim' ('pass'|'fail'|'none'), "
            "'dmarc' ('pass'|'fail'|'none')"
        )
    )

    class Config:
        json_schema_extra = {"example": {
            "body_text": "URGENT: Your account is suspended. Verify immediately.",
            "urls": ["http://paypa1-secure.example.tk/login"],
            "sender": "security@paypa1-alert.com",
            "reply_to": "verify@account-recovery.example",
            "headers": {"spf": "fail", "dkim": "fail", "dmarc": "fail"}
        }}


class SignalsDict(BaseModel):
    header_auth_failed: bool
    urgency_detected: bool
    suspicious_urls: int
    typosquatting_detected: bool


class PredictResponse(BaseModel):
    classification: str
    risk_score: float
    confidence: float
    risk_level: str
    model_type: str
    threshold_used: float
    signals: SignalsDict
    processing_time_ms: float
    escalated_to_tier2: bool = False
    tier2_verdict: Optional[str] = None
    tier2_confidence: Optional[float] = None
    tier2_xai_tokens: Optional[list] = None


class HealthResponse(BaseModel):
    status: str
    model_type: str
    model_loaded: bool
    distilbert_ready: bool
    thresholds: Dict[str, float]
    threshold_source: str
    error: Optional[str] = None


class RecalibrateRequest(BaseModel):
    suspicious_lower: float
    suspicious_upper: float
    phishing_threshold: float


class RecalibrateResponse(BaseModel):
    status: str
    new_thresholds: Dict[str, float]


# ---------------------------------------------------------------------------
# Signal extraction helpers
# ---------------------------------------------------------------------------
def _extract_signals(req: PredictRequest) -> dict:
    """Extract interpretable signals from the request for the response body."""
    from ml.features.url_features import extract as url_extract

    # Header auth failures
    hdrs = req.headers or {}
    spf  = str(hdrs.get("spf",  "none")).lower()
    dkim = str(hdrs.get("dkim", "none")).lower()
    dmarc= str(hdrs.get("dmarc","none")).lower()
    header_auth_failed = any(v in ("fail", "softfail") for v in [spf, dkim, dmarc])

    # Urgency keyword count
    body_lower = req.body_text.lower()
    urgency_count = sum(1 for kw in URGENCY_KEYWORDS if kw in body_lower)
    urgency_detected = urgency_count >= 2

    # Auto-extract URLs from body if not explicitly passed
    import re
    from ml.features.url_features import check_typosquatting
    all_urls = list(req.urls or [])
    if not all_urls and req.body_text:
        found_urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', req.body_text)
        all_urls.extend(found_urls)

    url_feats = url_extract(all_urls)
    suspicious_urls_count = int(
        url_feats["any_http_url"] +
        url_feats["any_phishing_tld"] +
        url_feats["any_typosquatting"]
    )
    typosquatting = bool(url_feats.get("any_typosquatting", 0))

    # Also check sender domain for typosquatting (e.g. service@paypa1-security.com)
    if not typosquatting and req.sender and "@" in req.sender:
        sender_domain = req.sender.split("@")[-1]
        t_flag, _ = check_typosquatting(sender_domain)
        if t_flag:
            typosquatting = True

    return {
        "header_auth_failed":    header_auth_failed,
        "urgency_detected":      urgency_detected,
        "suspicious_urls":       suspicious_urls_count,
        "typosquatting_detected": typosquatting,
    }


def _extract_header_features(req: PredictRequest) -> dict:
    """
    Parse headers dict into 10 boolean/int features for model input.
    Keys: spf_pass, spf_fail, spf_none, dkim_pass, dkim_fail, dkim_none,
          dmarc_pass, dmarc_fail, dmarc_none, sender_domain_match
    """
    from ml.features.header_features import extract as header_extract, HEADER_FEATURE_NAMES
    hdrs = req.headers or {}
    feat = header_extract(
        headers_dict=hdrs,
        sender=req.sender or "",
        reply_to=req.reply_to or "",
    )
    return feat


def _risk_level(score: float) -> str:
    if score < 0.08:   return "LOW"
    if score < 0.20:   return "MEDIUM"
    if score < 0.50:   return "HIGH"
    return "CRITICAL"


def _classify(risk_score: float, confidence: float) -> str:
    pt = state.phishing_threshold
    sl = state.suspicious_lower
    if risk_score >= pt:
        return "PHISHING"
    elif risk_score >= sl:
        return "SUSPICIOUS"
    else:
        return "LEGITIMATE"


# ---------------------------------------------------------------------------
# RF predict
# ---------------------------------------------------------------------------
def _predict_rf(req: PredictRequest) -> dict:
    from scipy.sparse import hstack, csr_matrix
    from ml.features.url_features import extract as url_extract, URL_FEATURE_NAMES
    from ml.features.header_features import extract as header_extract, HEADER_FEATURE_NAMES
    import pandas as pd

    df = pd.DataFrame([{
        "body_text":         req.body_text or "",
        "subject":           "",
        "urls":              json.dumps(req.urls or []),
        "sender":            req.sender or "",
        "reply_to":          req.reply_to or "",
        "headers_available": json.dumps(req.headers or {}),
    }])

    X_text = state.text_extractor.transform(df)

    url_feat = url_extract(req.urls or [])
    X_url    = csr_matrix(np.array([url_feat[k] for k in URL_FEATURE_NAMES],
                                    dtype=np.float32).reshape(1, -1))

    hdr_feat = header_extract(req.headers or {}, req.sender or "", req.reply_to or "")
    X_hdr    = csr_matrix(np.array([hdr_feat[k] for k in HEADER_FEATURE_NAMES],
                                    dtype=np.float32).reshape(1, -1))

    X          = hstack([X_text, X_url, X_hdr])
    proba      = state.rf_model.predict_proba(X)[0]
    risk_score = float(proba[1])
    pt = state.phishing_threshold
    sl = state.suspicious_lower

    if risk_score >= pt:
        cls        = "PHISHING"
        confidence = risk_score - pt
    elif risk_score >= sl:
        cls        = "SUSPICIOUS"
        confidence = min(risk_score - sl, pt - risk_score)
    else:
        cls        = "LEGITIMATE"
        confidence = sl - risk_score

    return {
        "classification": cls,
        "risk_score":     round(risk_score, 4),
        "confidence":     round(max(confidence, 0.0), 4),
        "threshold_used": round(pt, 4),
    }


# ---------------------------------------------------------------------------
# DistilBERT predict
# ---------------------------------------------------------------------------
def _predict_distilbert(req: PredictRequest) -> dict:
    import torch
    from ml.features.header_features import extract as header_extract, HEADER_FEATURE_NAMES

    max_len     = state.db_config.get("max_length", 256)
    conf_thresh = state.db_config.get("confidence_threshold", 0.70)

    # Format input text with Subject, Body, and URLs to match training dataset structure
    import re
    text_parts = []
    subject = getattr(req, 'subject', '') or ''
    if subject.strip():
        text_parts.append(f'Subject: {subject.strip()}')
    if req.body_text and req.body_text.strip():
        text_parts.append(f"Body: {req.body_text.strip()}")

    # Include detected URLs as signals in the prompt
    all_urls = list(req.urls or [])
    if req.body_text:
        all_urls.extend(re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', req.body_text))
    for u in set(all_urls):
        text_parts.append(f"Phishing URL: {u}")

    model_input_text = "\n".join(text_parts) if text_parts else (req.body_text or "")

    enc = state.db_tokenizer(
        model_input_text,
        max_length=max_len, truncation=True, padding="max_length", return_tensors="pt"
    )

    hdr_feat = header_extract(req.headers or {}, req.sender or "", req.reply_to or "")
    hdr_tensor = torch.tensor(
        [[float(hdr_feat.get(k, 0)) for k in HEADER_FEATURE_NAMES]], dtype=torch.float32
    )

    with torch.no_grad():
        logits = state.db_model(
            enc["input_ids"], enc["attention_mask"], hdr_tensor
        )
        proba      = torch.softmax(logits, dim=1)[0].numpy()
        risk_score = float(proba[1])
        confidence = float(max(proba))
        pt         = state.phishing_threshold

        # If model is uncertain or score falls into suspicious corridor
        if confidence < conf_thresh or (state.suspicious_lower <= risk_score < pt):
            cls = "SUSPICIOUS"
        elif proba[1] > proba[0] or risk_score >= pt:
            cls = "PHISHING"
        else:
            cls = "LEGITIMATE"

    # Multi-signal override: If deep learning scored low but multiple high-confidence
    # threat signals were detected (e.g. typosquatted domain + failed headers or malicious URLs)
    signals = _extract_signals(req)
    high_threat_signals = (
        int(signals.get("typosquatting_detected", False)) +
        int(signals.get("header_auth_failed", False)) +
        min(signals.get("suspicious_urls", 0), 2)
    )

    if cls == "LEGITIMATE" and high_threat_signals >= 2:
        cls = "SUSPICIOUS"
        risk_score = max(risk_score, 0.18)  # Elevate into suspicious corridor
    elif cls == "LEGITIMATE" and high_threat_signals >= 3:
        cls = "PHISHING"
        risk_score = max(risk_score, 0.75)

    return {
        "classification": cls,
        "risk_score":     round(risk_score, 4),
        "confidence":     round(confidence, 4),
        "threshold_used": round(pt, 4),
    }



# ---------------------------------------------------------------------------
# Tier-2 Escalation Logic
# ---------------------------------------------------------------------------
def should_escalate(tier1_result: dict, signals: dict) -> bool:
    """
    Decide if this email should be escalated to Tier-2 for deeper analysis.
    Uses production escalation policy (tighter than training corridor).
    """
    if not state.tier2_enabled:
        return False

    policy = state.tier2_config.get("escalation_policy", {})
    lower  = policy.get("suspicious_lower", 0.08)
    upper  = policy.get("suspicious_upper", 0.25)
    conf_thresh = policy.get("confidence_threshold", 0.70)

    verdict    = tier1_result["classification"]
    risk_score = tier1_result["risk_score"]
    confidence = tier1_result["confidence"]

    # Always escalate SUSPICIOUS
    if verdict == "SUSPICIOUS":
        return True

    # Escalate uncertain PHISHING
    if verdict == "PHISHING" and confidence < conf_thresh:
        return True

    # Escalate LEGITIMATE with conflicting threat signals
    if verdict == "LEGITIMATE":
        signal_count = (
            int(signals.get("typosquatting_detected", False)) +
            int(signals.get("header_auth_failed", False)) +
            min(signals.get("suspicious_urls", 0), 2)
        )
        if signal_count >= 1:
            return True

    return False

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    return HealthResponse(
        status="ok",
        model_type="distilbert",
        model_loaded=state.db_ready,
        distilbert_ready=state.db_ready,
        thresholds={
            "phishing":         round(state.phishing_threshold, 4),
            "suspicious_lower": round(state.suspicious_lower, 4),
            "suspicious_upper": round(state.suspicious_upper, 4),
        },
        threshold_source=state.threshold_source,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(req: PredictRequest):
    if not state.model_loaded:
        raise HTTPException(status_code=503, detail={
            "error": "model_not_loaded",
            "message": "No model loaded. Run ml/train.py first.",
        })
    if not req.body_text or not req.body_text.strip():
        raise HTTPException(status_code=422, detail="body_text must not be empty.")

    t0 = time.perf_counter()
    try:
        if state.db_ready:
            result = _predict_distilbert(req)
        else:
            result = _predict_rf(req)

        signals = _extract_signals(req)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)

        log.info(f"Predict [{state.model_type}]: {result['classification']} "
                 f"score={result['risk_score']:.4f} "
                 f"urgency={signals['urgency_detected']} "
                 f"typosquat={signals['typosquatting_detected']} "
                 f"auth_fail={signals['header_auth_failed']} "
                 f"elapsed={elapsed}ms")

        # Check for Tier-2 Escalation
        escalated = False
        t2_res = None
        if should_escalate(result, signals) and state.tier2_enabled and state.tier2_client:
            log.info(f"Escalating email to Tier-2 service (verdict={result['classification']}, score={result['risk_score']:.4f})...")
            t2_payload = {
                "subject": getattr(req, "subject", "") or "",
                "body_text": req.body_text,
                "urls": req.urls or [],
                "sender": req.sender or "",
                "reply_to": req.reply_to or "",
                "headers_available": req.headers or {},
                "tier1_risk_score": result["risk_score"],
                "tier1_confidence": result["confidence"],
                "tier1_verdict": result["classification"],
                "tier1_signals": signals,
            }
            t2_res = await state.tier2_client.predict(t2_payload)
            if t2_res:
                escalated = True
                log.info(f"Tier-2 resolved: verdict={t2_res.get('verdict')}, conf={t2_res.get('confidence')}")

        return PredictResponse(
            classification     = t2_res.get("verdict", result["classification"]) if t2_res else result["classification"],
            risk_score         = t2_res.get("risk_score", result["risk_score"]) if t2_res else result["risk_score"],
            confidence         = t2_res.get("confidence", result["confidence"]) if t2_res else result["confidence"],
            risk_level         = _risk_level(t2_res.get("risk_score", result["risk_score"]) if t2_res else result["risk_score"]),
            model_type         = "distilbert+roberta_tier2" if escalated else state.model_type,
            threshold_used     = result["threshold_used"],
            signals            = SignalsDict(**signals),
            processing_time_ms = elapsed,
            escalated_to_tier2 = escalated,
            tier2_verdict      = t2_res.get("verdict") if t2_res else None,
            tier2_confidence   = t2_res.get("confidence") if t2_res else None,
            tier2_xai_tokens   = [t.dict() if hasattr(t, 'dict') else t for t in t2_res.get("top_tokens", [])] if t2_res else None,
        )

    except Exception as e:
        log.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/admin/recalibrate", response_model=RecalibrateResponse, tags=["admin"])
async def recalibrate(req: RecalibrateRequest):
    """
    Live threshold tuning without server restart.
    Useful for tuning during mentor demos.

    Validates: 0 < suspicious_lower < suspicious_upper <= phishing_threshold < 1.0
    """
    sl = req.suspicious_lower
    su = req.suspicious_upper
    pt = req.phishing_threshold

    if not (0 < sl < su <= pt < 1.0):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid threshold ordering. Required: 0 < suspicious_lower({sl}) "
                   f"< suspicious_upper({su}) <= phishing_threshold({pt}) < 1.0"
        )

    state.suspicious_lower    = sl
    state.suspicious_upper    = su
    state.phishing_threshold  = pt
    state.threshold_source    = "manual_recalibrate"
    _save_thresholds()

    log.info(f"Thresholds recalibrated: phishing>={pt} | suspicious [{sl}, {su})")
    return RecalibrateResponse(
        status="recalibrated",
        new_thresholds={
            "phishing":         pt,
            "suspicious_lower": sl,
            "suspicious_upper": su,
        }
    )


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Not found", "path": str(request.url.path)},
    )