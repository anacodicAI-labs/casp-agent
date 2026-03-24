#!/usr/bin/env python3
"""
Environment variable checker for production deployment.
Validates that all required environment variables are set correctly.
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Try to load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"📄 Loaded .env file from: {env_path}")
        print()
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")
    print()

def check_aws_credentials() -> Tuple[bool, List[str]]:
    """Check if AWS credentials are available."""
    issues = []
    
    # Check environment variables
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
    
    # Check AWS credentials file
    aws_creds_file = os.path.expanduser("~/.aws/credentials")
    aws_config_file = os.path.expanduser("~/.aws/config")
    
    has_env_vars = access_key and secret_key
    has_creds_file = os.path.exists(aws_creds_file)
    
    if not has_env_vars and not has_creds_file:
        issues.append("❌ AWS credentials not found!")
        issues.append("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables")
        issues.append("   OR configure ~/.aws/credentials file")
        return False, issues
    
    if has_env_vars:
        issues.append("✅ AWS credentials found in environment variables")
    elif has_creds_file:
        issues.append("✅ AWS credentials file found at ~/.aws/credentials")
    
    # Test boto3 import and basic client creation
    try:
        import boto3
        region = os.environ.get("AWS_REGION", "us-east-1")
        try:
            client = boto3.client("bedrock-runtime", region_name=region)
            issues.append(f"✅ boto3 can create Bedrock client for region: {region}")
        except Exception as e:
            issues.append(f"⚠️  boto3 import OK but Bedrock client creation failed: {e}")
            issues.append("   This might be due to:")
            issues.append("   - Invalid AWS credentials")
            issues.append("   - Bedrock not available in the specified region")
            issues.append("   - Missing IAM permissions for Bedrock")
            return False, issues
    except ImportError:
        issues.append("❌ boto3 not installed! Run: pip install boto3")
        return False, issues
    
    return True, issues


def check_region() -> Tuple[bool, List[str]]:
    """Check AWS region configuration."""
    issues = []
    region = os.environ.get("AWS_REGION", "us-east-1")
    issues.append(f"✅ AWS_REGION: {region} (default: us-east-1)")
    return True, issues


def check_model_id() -> Tuple[bool, List[str]]:
    """Check Bedrock model ID configuration."""
    issues = []
    model_id = os.environ.get("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0")
    issues.append(f"✅ BEDROCK_MODEL_ID: {model_id}")
    return True, issues


def check_api_keys() -> Tuple[bool, List[str]]:
    """Check optional API keys."""
    issues = []
    
    keys = {
        "OPENWEATHERMAP_API_KEY": "Weather API (optional)",
        "NEWSAPI_API_KEY": "News API (optional)",
        "OPENROUTESERVICE_API_KEY": "Distance API (optional)",
    }
    
    for key, description in keys.items():
        value = os.environ.get(key, "")
        if value:
            issues.append(f"✅ {key}: Set ({description})")
        else:
            issues.append(f"⚠️  {key}: Not set ({description} - fallback methods will be used)")
    
    return True, issues


def check_feature_flags() -> Tuple[bool, List[str]]:
    """Check feature flag configuration."""
    issues = []
    
    fallback = os.environ.get("FALLBACK_ENABLED", "true").strip().lower() in ("1", "true", "yes")
    web_search = os.environ.get("WEB_SEARCH_ENABLED", "true").strip().lower() in ("1", "true", "yes")
    
    issues.append(f"✅ FALLBACK_ENABLED: {fallback}")
    issues.append(f"✅ WEB_SEARCH_ENABLED: {web_search}")
    
    return True, issues


def check_strands() -> Tuple[bool, List[str]]:
    """Check if strands-agents is installed."""
    issues = []
    try:
        import strands
        issues.append("✅ strands-agents: Installed")
    except ImportError:
        issues.append("❌ strands-agents: Not installed! Run: pip install strands-agents")
        return False, issues
    
    return True, issues


def main():
    """Run all checks."""
    print("=" * 60)
    print("Supply Chain System - Environment Variable Checker")
    print("=" * 60)
    print()
    
    all_ok = True
    all_issues = []
    
    # Required checks
    print("🔐 AWS Credentials (REQUIRED):")
    print("-" * 60)
    ok, issues = check_aws_credentials()
    all_ok = all_ok and ok
    all_issues.extend(issues)
    print("\n".join(issues))
    print()
    
    print("🌍 AWS Region Configuration:")
    print("-" * 60)
    ok, issues = check_region()
    all_issues.extend(issues)
    print("\n".join(issues))
    print()
    
    print("🤖 Bedrock Model Configuration:")
    print("-" * 60)
    ok, issues = check_model_id()
    all_issues.extend(issues)
    print("\n".join(issues))
    print()
    
    print("📦 Dependencies:")
    print("-" * 60)
    ok, issues = check_strands()
    all_ok = all_ok and ok
    all_issues.extend(issues)
    print("\n".join(issues))
    print()
    
    # Optional checks
    print("🔑 External API Keys (OPTIONAL):")
    print("-" * 60)
    ok, issues = check_api_keys()
    all_issues.extend(issues)
    print("\n".join(issues))
    print()
    
    print("⚙️  Feature Flags:")
    print("-" * 60)
    ok, issues = check_feature_flags()
    all_issues.extend(issues)
    print("\n".join(issues))
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✅ All required checks passed!")
        print("   Your environment is configured correctly for production.")
        return 0
    else:
        print("❌ Some required checks failed!")
        print("   Please fix the issues above before deploying to production.")
        print()
        print("Quick fix:")
        print("  1. Set AWS credentials: export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...")
        print("  2. Set AWS region: export AWS_REGION=us-east-1")
        print("  3. Install dependencies: pip install boto3 strands-agents")
        return 1


if __name__ == "__main__":
    sys.exit(main())
