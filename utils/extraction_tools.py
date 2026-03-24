"""
Extraction tools: parse / normalize user input into structured features.
Uses LLM's own knowledge to extract and fill realistic values - no hardcoded rules.
"""

import os
import re
import json
from typing import Dict, Any, List, Tuple, Optional

VALID_PACKAGE_TYPES = [
    "pharmacy", "clothing", "electronics", "groceries",
    "automobile parts", "fragile items", "documents", "furniture", "cosmetics"
]
VALID_VEHICLE_TYPES = ["bike", "ev bike", "scooter", "ev van", "van", "truck"]
VALID_DELIVERY_MODES = ["same day", "express", "two day", "standard"]
VALID_WEATHER_CONDITIONS = ["clear", "cold", "rainy", "foggy", "hot", "stormy"]
VALID_REGIONS = ["west", "central", "east", "north", "south"]
VALID_PARTNERS = [
    "delhivery", "xpressbees", "shadowfax", "dhl", "amazon logistics",
    "blue dart", "fedex", "ecom express", "ekart", "dtdc", "gati"
]

PACKAGE_TYPE_SYNONYMS: Dict[str, str] = {
    "fashion": "clothing", "apparel": "clothing", "clothes": "clothing",
    "medicine": "pharmacy", "medical": "pharmacy", "pharma": "pharmacy",
    "insulin": "pharmacy", "vaccine": "pharmacy", "vaccines": "pharmacy",
    "drugs": "pharmacy", "medication": "pharmacy", "prescription": "pharmacy",
    "food": "groceries", "perishables": "groceries", "grocery": "groceries",
    "fragile": "fragile items", "breakable": "fragile items", "glass": "fragile items",
    "car parts": "automobile parts", "auto": "automobile parts", "auto parts": "automobile parts",
    "electronic": "electronics", "docs": "documents",
}

SMART_EXTRACTION_PROMPT = """You are a logistics expert with deep knowledge of Indian geography, transportation, and supply chain operations.

USER QUERY: "{query}"

Extract shipment details from the query. For anything NOT explicitly mentioned, use YOUR OWN KNOWLEDGE to fill realistic, practical values.

Think about:
- Actual road distances between Indian cities (you know this)
- What vehicle types can realistically handle this weight and distance
- How long deliveries actually take over different distances
- Which carriers operate on which routes in India
- What region each Indian city belongs to

Return ONLY a JSON object with these keys:
{{
  "package_type": "one of: pharmacy, clothing, electronics, groceries, automobile parts, fragile items, documents, furniture, cosmetics",
  "origin": "city name",
  "destination": "city name",
  "distance_km": <realistic distance based on your geography knowledge>,
  "package_weight_kg": <from query or realistic estimate>,
  "vehicle_type": "one of: bike, ev bike, scooter, ev van, van, truck",
  "delivery_mode": "one of: same day, express, two day, standard",
  "delivery_partner": "one of: delhivery, xpressbees, shadowfax, dhl, blue dart, fedex, ecom express, ekart, dtdc, gati",
  "weather_condition": "one of: clear, cold, rainy, foggy, hot, stormy",
  "region": "one of: west, central, east, north, south"
}}

Use your knowledge. Be practical. Return JSON only."""


def extract_with_smart_llm(query: str, region_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not query or not query.strip():
        return None

    region_name = region_name or os.environ.get("AWS_REGION", "us-east-1")
    model_id = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")

    try:
        import boto3
        client = boto3.client("bedrock-runtime", region_name=region_name)
        prompt = SMART_EXTRACTION_PROMPT.format(query=query.strip())
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
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
        if not text:
            return None
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            features = json.loads(match.group())
            return _validate_and_normalize_features(features)
        features = json.loads(text)
        return _validate_and_normalize_features(features)
    except Exception as e:
        print(f"[extraction_tools] LLM extraction failed: {e}")
        return None


def _validate_and_normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    result = {}
    pt = str(features.get("package_type", "clothing")).lower().strip()
    if pt in PACKAGE_TYPE_SYNONYMS:
        pt = PACKAGE_TYPE_SYNONYMS[pt]
    if pt not in VALID_PACKAGE_TYPES:
        pt = "clothing"
    result["package_type"] = pt
    result["origin"] = str(features.get("origin", "mumbai")).lower().strip()
    result["destination"] = str(features.get("destination", "delhi")).lower().strip()
    try:
        result["distance_km"] = float(features.get("distance_km", 500))
    except (TypeError, ValueError):
        result["distance_km"] = 500.0
    try:
        result["package_weight_kg"] = float(features.get("package_weight_kg", 10))
    except (TypeError, ValueError):
        result["package_weight_kg"] = 10.0
    vt = str(features.get("vehicle_type", "van")).lower().strip()
    if vt not in VALID_VEHICLE_TYPES:
        vt = "van"
    result["vehicle_type"] = vt
    dm = str(features.get("delivery_mode", "standard")).lower().strip()
    if dm not in VALID_DELIVERY_MODES:
        dm = "standard"
    result["delivery_mode"] = dm
    dp = str(features.get("delivery_partner", "delhivery")).lower().strip()
    if dp not in VALID_PARTNERS:
        dp = "delhivery"
    result["delivery_partner"] = dp
    wc = str(features.get("weather_condition", "clear")).lower().strip()
    if wc not in VALID_WEATHER_CONDITIONS:
        wc = "clear"
    result["weather_condition"] = wc
    rg = str(features.get("region", "north")).lower().strip()
    if rg not in VALID_REGIONS:
        rg = "north"
    result["region"] = rg
    result["delayed"] = "no"
    result["delivery_status"] = "delivered"
    result["delivery_rating"] = 4
    return result


def _infer_package_type_and_values(query: str) -> Dict[str, Any]:
    q = query.lower().strip()
    extracted: Dict[str, Any] = {}
    for phrase, canonical in sorted(PACKAGE_TYPE_SYNONYMS.items(), key=lambda x: -len(x[0])):
        if phrase in q:
            extracted["package_type"] = canonical
            break
    if "package_type" not in extracted:
        extracted["package_type"] = "clothing"
    dist_match = re.search(r"\b(\d+(?:\.\d+)?)\s*km\b", q, re.IGNORECASE)
    if dist_match:
        try:
            extracted["distance_km"] = float(dist_match.group(1))
        except (ValueError, TypeError):
            pass
    weight_match = re.search(r"\b(\d+(?:\.\d+)?)\s*kg\b", q, re.IGNORECASE)
    if weight_match:
        try:
            extracted["package_weight_kg"] = float(weight_match.group(1))
        except (ValueError, TypeError):
            pass
    indian_cities = ["mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "pune", "ahmedabad", "jaipur", "lucknow", "surat", "nagpur"]
    found_cities = [c for c in indian_cities if c in q]
    if len(found_cities) >= 2:
        extracted["origin"] = found_cities[0]
        extracted["destination"] = found_cities[1]
    elif len(found_cities) == 1:
        extracted["destination"] = found_cities[0]
    return extracted


def enrich_extracted_features(features: Dict[str, Any], query: str) -> Tuple[Dict[str, Any], List[str]]:
    """
    Apply geography hints, optional live weather, vehicle heuristics, and defaults_used tagging
    to an already-extracted feature dict (used by smart LLM path and streaming gather refinement).
    """
    features = dict(features)
    origin = str(features.get("origin", "") or "")
    destination = str(features.get("destination", "") or "")
    try:
        distance_km = float(features.get("distance_km", 0) or 0)
    except (TypeError, ValueError):
        distance_km = 0.0
    origin_city = None
    dest_city = None
    try:
        from config.routes import get_city

        origin_city = get_city(origin) if origin else None
        dest_city = get_city(destination) if destination else None
        if origin_city:
            features["region"] = origin_city.region
        elif dest_city:
            features["region"] = dest_city.region
    except ImportError:
        pass
    if not origin_city and origin:
        region_hints = {
            "west bengal": "east", "bengal": "east", "odisha": "east", "orissa": "east",
            "bihar": "east", "jharkhand": "east", "assam": "east", "tripura": "east",
            "maharashtra": "west", "gujarat": "west", "goa": "west", "rajasthan": "west",
            "karnataka": "south", "tamil nadu": "south", "kerala": "south", "andhra": "south",
            "telangana": "south", "puducherry": "south",
            "punjab": "north", "haryana": "north", "himachal": "north", "uttarakhand": "north",
            "uttar pradesh": "north", "delhi": "north", "jammu": "north", "kashmir": "north",
            "madhya pradesh": "central", "chhattisgarh": "central",
        }
        origin_lower = origin.lower()
        for state, region in region_hints.items():
            if state in origin_lower:
                features["region"] = region
                break
    if origin and destination:
        try:
            from config.api_config import API_KEYS, WEB_SEARCH_ENABLED

            weather = None
            owm_key = (API_KEYS or {}).get("openweathermap", "")
            if owm_key and owm_key != "your_openweathermap_api_key":
                try:
                    from data.apis.weather_api import get_weather

                    weather = get_weather(origin, owm_key) or get_weather(destination, owm_key)
                except Exception:
                    pass
            if not weather and WEB_SEARCH_ENABLED:
                try:
                    from utils.web_search import web_search, parse_weather_from_text

                    raw = web_search(f"{origin} weather today", enabled=True)
                    weather = parse_weather_from_text(raw) if raw else None
                    if not weather:
                        raw = web_search(f"{destination} weather today", enabled=True)
                        weather = parse_weather_from_text(raw) if raw else None
                except Exception as e:
                    print(f"[extraction_tools] Web search for weather failed: {e}")
            if weather and weather in VALID_WEATHER_CONDITIONS:
                features["weather_condition"] = weather
        except ImportError:
            pass
    if distance_km > 1000 and features.get("vehicle_type") == "van":
        features["vehicle_type"] = "truck"
    elif distance_km > 500 and features.get("vehicle_type") in ["bike", "ev bike", "scooter"]:
        features["vehicle_type"] = "van"
    q = (query or "").lower()
    explicit_keys = set()
    if any(pt in q for pt in VALID_PACKAGE_TYPES) or any(syn in q for syn in PACKAGE_TYPE_SYNONYMS):
        explicit_keys.add("package_type")
    if re.search(r"\d+\s*km", q):
        explicit_keys.add("distance_km")
    if re.search(r"\d+\s*kg", q):
        explicit_keys.add("package_weight_kg")
    indian_cities = [
        "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad",
        "pune", "ahmedabad", "jaipur", "lucknow",
    ]
    found_cities = [c for c in indian_cities if c in q]
    if len(found_cities) >= 1:
        explicit_keys.add("origin")
        explicit_keys.add("destination")
    defaults_used = [k for k in features.keys() if k not in explicit_keys]
    return features, defaults_used


def extract_from_query_and_merge_defaults(query: str, use_smart_llm: bool = True) -> Tuple[Dict[str, Any], List[str]]:
    if use_smart_llm:
        features = extract_with_smart_llm(query)
        if features:
            return enrich_extracted_features(features, query)

    from utils.defaults_from_data import get_defaults_for_package
    extracted = _infer_package_type_and_values(query)
    package_type = extracted.get("package_type", "clothing")
    defaults = get_defaults_for_package(package_type)
    features: Dict[str, Any] = dict(defaults)
    features.setdefault("origin", "mumbai")
    features.setdefault("destination", "delhi")
    for key, val in extracted.items():
        if val is not None and val != "":
            features[key] = val
    features = extract_features_from_dict(features)
    defaults_used = [k for k in features.keys() if k not in extracted]
    return features, defaults_used


def extract_features_from_dict(raw: Dict[str, Any]) -> Dict[str, Any]:
    features = dict(raw)
    if "package_type" in features and features["package_type"]:
        features["package_type"] = str(features["package_type"]).lower().strip()
    for key in ("origin", "destination"):
        if key in features and features[key]:
            features[key] = str(features[key]).lower().strip()
    for key in ("distance_km", "package_weight_kg"):
        if key in features and features[key] is not None:
            try:
                features[key] = float(features[key])
            except (TypeError, ValueError):
                pass
    return features

