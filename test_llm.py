#!/usr/bin/env python3
"""
Simple test script to verify LLM model calls are working
"""
import asyncio
import os
from kaggle_ts.config import code_llm_provider, idea_llm_provider
from kaggle_ts.providers.gemini import GeminiProvider, GeminiFlashProvider, GeminiProProvider

async def test_llm_calls():
    print("🔍 Testing LLM Model Calls")
    print("=" * 50)

    # Test 1: Check environment variables
    print("1. Environment Check:")
    gemini_key = os.getenv('GEMINI_API_KEY')
    if gemini_key:
        print(f"   ✅ GEMINI_API_KEY is set (length: {len(gemini_key)})")
    else:
        print("   ❌ GEMINI_API_KEY not found")
        return

    # Test 2: Check code LLM provider from config
    print("\n2. Code LLM Provider (from config):")
    try:
        code_provider = code_llm_provider()
        print(f"   ✅ Code provider loaded: {code_provider.__class__.__name__}")
        print(f"   ✅ Model: {getattr(code_provider, '_model', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Code provider failed: {e}")
        import traceback
        print(f"   🔍 Full traceback: {traceback.format_exc()}")
        return

    # Test 3: Check idea LLM provider from config
    print("\n3. Idea LLM Provider (from config):")
    try:
        idea_provider = idea_llm_provider()
        print(f"   ✅ Idea provider loaded: {idea_provider.__class__.__name__}")
        print(f"   ✅ Model: {getattr(idea_provider, '_model', 'unknown')}")
    except Exception as e:
        print(f"   ❌ Idea provider failed: {e}")
        import traceback
        print(f"   🔍 Full traceback: {traceback.format_exc()}")
        return

    # Test 4: Simple code generation test
    print("\n4. Code Generation Test (using config provider):")
    test_prompt = "Write a simple Python function that adds two numbers."
    try:
        print(f"   📝 Prompt: {test_prompt}")
        response = await code_provider.generate(test_prompt, max_tokens=3072)
        if response and len(response.strip()) > 10:
            print("   ✅ Code generation successful!")
            print(f"   📄 Response length: {len(response)} characters")
            print("   📄 Response preview:")
            print("   " + "\n   ".join(response.split('\n')[:3]))
        else:
            print("   ❌ Code generation returned empty or invalid response")
            print(f"   🔍 Actual response: {repr(response)}")
    except Exception as e:
        print(f"   ❌ Code generation failed: {e}")
        import traceback
        print(f"   🔍 Full traceback: {traceback.format_exc()}")

    # Test 5: Simple idea generation test
    print("\n5. Idea Generation Test (using config provider):")
    test_challenge = "Predict Titanic survival based on passenger data."
    try:
        prompt = f"Generate one strategy for: {test_challenge}"
        print(f"   📝 Prompt: {prompt}")
        response = await idea_provider.generate(prompt, max_tokens=3072)
        if response and len(response.strip()) > 10:
            print("   ✅ Idea generation successful!")
            print(f"   📄 Response length: {len(response)} characters")
            print("   📄 Response preview:")
            print("   " + "\n   ".join(response.split('\n')[:2]))
        else:
            print("   ❌ Idea generation returned empty or invalid response")
            print(f"   🔍 Actual response: {repr(response)}")
    except Exception as e:
        print(f"   ❌ Idea generation failed: {e}")
        import traceback
        print(f"   🔍 Full traceback: {traceback.format_exc()}")

    # Test 6: Directly test specific Gemini providers
    print("\n6. Direct Provider Test (Pro vs Flash):")
    try:
        prompt = "Write a simple Python function to multiply two numbers."
        print(f"   📝 Prompt: {prompt}")

        print("\n   Testing GeminiProProvider...")
        pro_provider = GeminiProProvider(temperature=0.7, max_output_tokens=3072)
        pro_resp = await pro_provider.generate(prompt)
        if pro_resp and len(pro_resp.strip()) > 10:
            print(f"   ✅ Pro response (first line): {pro_resp.splitlines()[0]}")
        else:
            print(f"   ❌ Pro provider returned invalid response: {repr(pro_resp)}")

        print("\n   Testing GeminiFlashProvider...")
        flash_provider = GeminiFlashProvider(temperature=0.7, max_output_tokens=256)
        flash_resp = await flash_provider.generate(prompt)
        if flash_resp and len(flash_resp.strip()) > 10:
            print(f"   ✅ Flash response (first line): {flash_resp.splitlines()[0]}")
        else:
            print(f"   ❌ Flash provider returned invalid response: {repr(flash_resp)}")

    except Exception as e:
        print(f"   ❌ Direct provider test failed: {e}")
        import traceback
        print(f"   🔍 Full traceback: {traceback.format_exc()}")

    print("\n" + "=" * 50)
    print("🎉 LLM testing complete!")

if __name__ == "__main__":
    asyncio.run(test_llm_calls())
