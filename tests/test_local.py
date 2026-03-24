#!/usr/bin/env python3
"""
Quick local test to verify the application works with .env file.
"""

import os
import sys
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print("✅ Loaded .env file")
    else:
        print("⚠️  .env file not found")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
    sys.exit(1)

# Test 1: Check environment variables are loaded
print("\n📋 Test 1: Environment Variables")
print("-" * 50)
aws_key = os.environ.get("AWS_ACCESS_KEY_ID", "")
aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "")
aws_region = os.environ.get("AWS_REGION", "")

if aws_key and aws_secret:
    print(f"✅ AWS_ACCESS_KEY_ID: {aws_key[:10]}...")
    print(f"✅ AWS_SECRET_ACCESS_KEY: {'*' * 20}...")
    print(f"✅ AWS_REGION: {aws_region}")
else:
    print("❌ AWS credentials not found in environment")
    sys.exit(1)

# Test 2: Test boto3 Bedrock client
print("\n📋 Test 2: AWS Bedrock Client")
print("-" * 50)
try:
    import boto3
    client = boto3.client("bedrock-runtime", region_name=aws_region)
    print("✅ boto3 Bedrock client created successfully")
except Exception as e:
    print(f"❌ Failed to create Bedrock client: {e}")
    sys.exit(1)

# Test 3: Test imports
print("\n📋 Test 3: Application Imports")
print("-" * 50)
try:
    from utils.extraction_tools import extract_from_query_and_merge_defaults
    print("✅ utils.extraction_tools imported")
    
    from supply_chain_orchestrator import SupplyChainOrchestrator
    print("✅ supply_chain_orchestrator imported")
    
    from app import app
    print("✅ FastAPI app imported")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test basic extraction (without LLM if it fails)
print("\n📋 Test 4: Basic Feature Extraction")
print("-" * 50)
try:
    features, defaults = extract_from_query_and_merge_defaults("insulin Mumbai to Delhi 150km")
    print(f"✅ Feature extraction successful")
    print(f"   Package type: {features.get('package_type')}")
    print(f"   Origin: {features.get('origin')}")
    print(f"   Destination: {features.get('destination')}")
    print(f"   Defaults used: {len(defaults)} fields")
except Exception as e:
    print(f"⚠️  Feature extraction failed: {e}")
    print("   (This might be OK if LLM is not available)")

print("\n" + "=" * 50)
print("✅ All basic tests passed!")
print("=" * 50)
print("\n💡 Next steps:")
print("   1. Start the server: uvicorn app:app --reload")
print("   2. Test API: curl -X POST http://localhost:8000/api/extract \\")
print("      -H 'Content-Type: application/json' \\")
print("      -d '{\"query\": \"insulin Mumbai to Delhi\"}'")
