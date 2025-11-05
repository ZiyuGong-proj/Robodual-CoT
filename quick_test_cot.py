#!/usr/bin/env python3
"""
Quick test script to verify CoT implementation without running full evaluation.
快速测试脚本,用于验证CoT实现而无需运行完整评估。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

def test_cot_functionality(model_path: str):
    """Test CoT functionality with a simple forward pass."""

    print("="*80)
    print("Testing CoT Implementation")
    print("="*80)

    # Load model
    print("\n1. Loading model...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).cuda()
        model.eval()
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Error loading model: {e}")
        return False

    # Create dummy input
    print("\n2. Creating test input...")
    try:
        # Create a dummy RGB image
        dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        instruction = "pick up the red block"

        # Test prompt without CoT
        from vla_scripts.dual_sys_evaluation import get_openvla_prompt
        prompt_no_cot = get_openvla_prompt(instruction, enable_cot=False)
        prompt_with_cot = get_openvla_prompt(instruction, enable_cot=True)

        print(f"   Prompt (no CoT): {prompt_no_cot}")
        print(f"   Prompt (CoT):    {prompt_with_cot}")
        print("   ✓ Test input created")
    except Exception as e:
        print(f"   ✗ Error creating input: {e}")
        return False

    # Test without CoT
    print("\n3. Testing without CoT...")
    try:
        inputs = processor(prompt_no_cot, dummy_image).to('cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            result = model.predict_action(
                enable_cot=False,
                **inputs
            )
        actions, hidden_states = result
        print(f"   Actions shape: {actions.shape}")
        print(f"   Hidden states shape: {hidden_states.shape if hidden_states is not None else 'None'}")
        print("   ✓ No-CoT inference successful")
    except Exception as e:
        print(f"   ✗ Error in no-CoT inference: {e}")
        return False

    # Test with CoT
    print("\n4. Testing with CoT...")
    try:
        inputs = processor(prompt_with_cot, dummy_image).to('cuda', dtype=torch.bfloat16)
        with torch.no_grad():
            result = model.predict_action(
                enable_cot=True,
                max_cot_tokens=50,
                **inputs
            )
        actions, hidden_states, cot_token_ids = result

        print(f"   Actions shape: {actions.shape}")
        print(f"   Hidden states shape: {hidden_states.shape if hidden_states is not None else 'None'}")

        if cot_token_ids is not None:
            cot_text = processor.tokenizer.decode(cot_token_ids, skip_special_tokens=True)
            print(f"   CoT tokens: {len(cot_token_ids)}")
            print(f"   CoT text preview: {cot_text[:100]}...")
            print("   ✓ CoT inference successful")
        else:
            print("   ⚠ No CoT tokens generated")

    except Exception as e:
        print(f"   ✗ Error in CoT inference: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quick_test_cot.py <path_to_generalist_model>")
        print("Example: python quick_test_cot.py /path/to/openvla-7b")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    success = test_cot_functionality(model_path)
    sys.exit(0 if success else 1)
