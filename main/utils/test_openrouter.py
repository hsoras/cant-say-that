#!/usr/bin/env python3
"""
Quick smoke test for OpenRouter models configured in .env.
Sends a trivial prompt to each role (attacker, judge, target) and reports results.
"""
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent.parent / ".env", override=False)

API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")

MODELS = {
    "attacker": os.environ.get("OPENROUTER_ATTACKER_MODEL", "anthropic/claude-sonnet-4"),
    "judge": os.environ.get("OPENROUTER_JUDGE_MODEL", "openai/gpt-4o-mini"),
    "target": os.environ.get("OPENROUTER_TARGET_MODEL", "meta-llama/llama-3.1-8b-instruct"),
}

TEST_PROMPT = "Reply with exactly: HELLO WORLD"


def test_model(role: str, model: str) -> bool:
    print(f"\n{'─'*50}")
    print(f"🔧 Testing {role.upper()} → {model}")
    print(f"{'─'*50}")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "stream": False,
        "max_tokens": 50,
    }

    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=headers,
            json=body,
            timeout=60,
        )
        elapsed = time.perf_counter() - t0
    except requests.RequestException as e:
        elapsed = time.perf_counter() - t0
        print(f"   ❌ NETWORK ERROR after {elapsed:.1f}s: {e}")
        return False

    print(f"   ⏱  Response time: {elapsed:.1f}s")
    print(f"   📡 HTTP status: {resp.status_code}")

    try:
        data = resp.json()
    except ValueError:
        print(f"   ❌ Invalid JSON: {resp.text[:200]}")
        return False

    if resp.status_code != 200:
        err = data.get("error", {})
        msg = err.get("message", resp.text[:200]) if isinstance(err, dict) else str(err)[:200]
        print(f"   ❌ API ERROR: {msg}")
        return False

    choices = data.get("choices", [])
    if not choices:
        print(f"   ❌ No choices in response")
        return False

    content = choices[0].get("message", {}).get("content", "")
    usage = data.get("usage", {})
    print(f"   📝 Response: {content.strip()[:100]}")
    print(f"   📊 Tokens: prompt={usage.get('prompt_tokens', '?')}, completion={usage.get('completion_tokens', '?')}")
    print(f"   ✅ OK")
    return True


def main():
    if not API_KEY:
        print("❌ OPENROUTER_API_KEY is not set in .env")
        sys.exit(1)

    print(f"🔑 API Key: {API_KEY[:12]}...{API_KEY[-4:]}")
    print(f"🌐 Base URL: {BASE_URL}")

    results = {}
    for role, model in MODELS.items():
        results[role] = test_model(role, model)

    print(f"\n{'='*50}")
    print("📋 SUMMARY")
    print(f"{'='*50}")
    all_ok = True
    for role, ok in results.items():
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"   {role.upper():10s} ({MODELS[role]}): {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n🎉 All models working!")
    else:
        print("\n⚠️  Some models failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
