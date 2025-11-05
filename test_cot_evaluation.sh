#!/bin/bash
# CoT Evaluation Test Script for Robodual
# 测试脚本:使用CoT评估Robodual系统

echo "=========================================="
echo "Robodual CoT Evaluation Test"
echo "=========================================="

# 配置路径 / Configure paths
GENERALIST_PATH="${1:-/path/to/your/generalist/model}"
SPECIALIST_PATH="${2:-/path/to/your/specialist_policy.pt}"

echo ""
echo "Configuration:"
echo "  Generalist Path: $GENERALIST_PATH"
echo "  Specialist Path: $SPECIALIST_PATH"
echo ""

# 检查路径是否存在 / Check if paths exist
if [ ! -d "$GENERALIST_PATH" ] && [ ! -f "$GENERALIST_PATH" ]; then
    echo "Error: Generalist path does not exist: $GENERALIST_PATH"
    echo "Usage: ./test_cot_evaluation.sh <generalist_path> <specialist_path>"
    exit 1
fi

if [ ! -f "$SPECIALIST_PATH" ]; then
    echo "Error: Specialist path does not exist: $SPECIALIST_PATH"
    echo "Usage: ./test_cot_evaluation.sh <generalist_path> <specialist_path>"
    exit 1
fi

cd vla-scripts

echo "=========================================="
echo "Test 1: Baseline (Without CoT)"
echo "=========================================="
echo ""

python evaluate_calvin.py \
    --generalist_path "$GENERALIST_PATH" \
    --specialist_path "$SPECIALIST_PATH" \
    --with_depth \
    --with_gripper

echo ""
echo "=========================================="
echo "Test 2: With CoT (max_cot_tokens=50)"
echo "=========================================="
echo ""

python evaluate_calvin.py \
    --generalist_path "$GENERALIST_PATH" \
    --specialist_path "$SPECIALIST_PATH" \
    --with_depth \
    --with_gripper \
    --enable_cot \
    --max_cot_tokens 50

echo ""
echo "=========================================="
echo "Test 3: With CoT (max_cot_tokens=100)"
echo "=========================================="
echo ""

python evaluate_calvin.py \
    --generalist_path "$GENERALIST_PATH" \
    --specialist_path "$SPECIALIST_PATH" \
    --with_depth \
    --with_gripper \
    --enable_cot \
    --max_cot_tokens 100

echo ""
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo ""
echo "Compare the results to see the impact of CoT on:"
echo "  1. Task success rate"
echo "  2. Inference latency (TTFT and TPOT)"
echo "  3. Reasoning quality (from printed CoT)"
