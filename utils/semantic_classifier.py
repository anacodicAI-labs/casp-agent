"""
Semantic package_type classifier using AWS Bedrock (Claude).
Used when rule-based extraction is ambiguous or would default to clothing.
Critical for life-safety: e.g. "insulin" -> pharmacy, not clothing.
"""

from typing import Optional
import os
import json
import re

VALID_PACKAGE_TYPES = {
    "pharmacy", "clothing", "electronics", "groceries",
    "automobile parts", "fragile items", "documents", "furniture", "cosmetics",
}

CLASSIFIER_SYSTEM = """You are a supply-chain logistics classifier. Given a short user query describing a shipment or product, output ONLY a JSON object with exactly one key "package_type" whose value is one of: pharmacy, clothing, electronics, groceries, automobile parts, fragile items, documents, furniture, cosmetics.

Rules:
- Medicine, drugs, vaccines, insulin, refrigerated drugs, medical supplies, prescriptions -> pharmacy
- Apparel, fashion, clothes, shoes -> clothing
- Phones, laptops, devices, gadgets -> electronics
- Food, perishables, fresh produce -> groceries
- Car parts, auto parts -> automobile parts
- Glass, breakables, ceramics -> fragile items
- Legal docs, contracts, papers -> documents
- Tables, sofas, large home items -> furniture
- Beauty, skincare, makeup -> cosmetics
- If truly ambiguous, prefer pharmacy over clothing for anything health-related.
Output nothing but the JSON object, no markdown or explanation."""


def classify_package_type_bedrock(query: str, region_name: Optional[str] = None, model_id: Optional[str] = None) -> Optional[str]:
    if not query or not query.strip():
        return None
    region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
    model_id = model_id or os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=region_name)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 128,
            "temperature": 0,
            "system": CLASSIFIER_SYSTEM,
            "messages": [{"role": "user", "content": query.strip()[:500]}],
        }
        response = client.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        raw = response.get("body").read()
        out = json.loads(raw)
        text = ""
        for block in out.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        if not text and isinstance(out.get("output", {}).get("message", {}).get("content"), list):
            for block in out["output"]["message"]["content"]:
                if block.get("type") == "text":
                    text += block.get("text", "")
        if not text:
            return None
        text = text.strip()
        m = re.search(r"\{[^{}]*\"package_type\"[^{}]*\}", text)
        if m:
            obj = json.loads(m.group())
            pt = (obj.get("package_type") or "").strip().lower()
            if pt in VALID_PACKAGE_TYPES:
                return pt
            if pt in ("fragile items", "fragile"):
                return "fragile items"
            if pt in ("automobile parts", "auto parts"):
                return "automobile parts"
        return None
    except Exception:
        return None

