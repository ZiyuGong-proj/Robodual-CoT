# Chain-of-Thought (CoT) Implementation for Robodual

## 概述 / Overview

本实现通过Zero-shot提示方法为Robodual系统的generalist模型添加了Chain-of-Thought (CoT)推理能力。Generalist会在生成action tokens之前生成思考过程,并在控制台打印出来,而specialist只接收action tokens对应的hidden states,不会受到CoT内容的影响。

This implementation adds Chain-of-Thought (CoT) reasoning capability to the Robodual system's generalist model using zero-shot prompting. The generalist generates reasoning before action tokens and prints it to console, while the specialist only receives hidden states corresponding to action tokens, unaffected by CoT content.

## 修改的文件 / Modified Files

1. **`prismatic/extern/hf/modeling_prismatic.py`**
   - 修改了 `OpenVLAForActionPrediction.predict_action()` 方法
   - 添加了 `enable_cot` 和 `max_cot_tokens` 参数
   - 增加了生成长度以容纳CoT + action tokens
   - 分离CoT tokens和action tokens
   - 只将action tokens的hidden states传递给specialist

2. **`vla-scripts/dual_sys_evaluation.py`**
   - 修改了 `get_openvla_prompt()` 函数,添加CoT触发提示词
   - 更新了 `DualSystemCalvinEvaluation.__init__()` 以支持CoT配置
   - 修改了 `_worker_loop()` 来处理和打印CoT输出
   - 在 `step()` 方法中传递CoT标志

3. **`vla-scripts/evaluate_calvin.py`**
   - 添加了命令行参数 `--enable_cot` 和 `--max_cot_tokens`
   - 将CoT参数传递给 `DualSystemCalvinEvaluation`

## 使用方法 / Usage

### 基本用法 / Basic Usage

**不启用CoT (默认行为):**
```bash
cd vla-scripts
python evaluate_calvin.py \
    --generalist_path /path/to/generalist/model \
    --specialist_path /path/to/specialist/policy.pt
```

**启用CoT:**
```bash
cd vla-scripts
python evaluate_calvin.py \
    --generalist_path /path/to/generalist/model \
    --specialist_path /path/to/specialist/policy.pt \
    --enable_cot \
    --max_cot_tokens 100
```

### 参数说明 / Parameters

- `--enable_cot`: 启用Chain-of-Thought推理 (默认: False)
- `--max_cot_tokens`: CoT推理的最大token数量 (默认: 100)

## 技术细节 / Technical Details

### CoT Prompt 设计

**Without CoT:**
```
In: What action should the robot take to {instruction}?
Out:
```

**With CoT:**
```
In: What action should the robot take to {instruction}? Let's think step by step.
Out:
```

添加 "Let's think step by step." 是经典的zero-shot CoT触发提示词。

### Token 序列结构

启用CoT后,生成的token序列结构为:
```
[Input tokens] [CoT reasoning tokens] [Action tokens (56 tokens)]
                    ↓                           ↓
              打印到控制台                传递给specialist
              (Printed to console)      (Passed to specialist)
```

### Hidden States 处理

- **Generalist输出**: 生成 `max_cot_tokens + action_dim` 个tokens
- **分离逻辑**:
  - CoT tokens: `generated_tokens[:-action_dim]`
  - Action tokens: `generated_tokens[-action_dim:]`
- **传递给Specialist**: 只有最后 `action_dim` 个tokens对应的hidden states被传递给specialist
- **结果**: Specialist不会受到CoT内容的影响,只接收纯粹的action信息

### 输出示例 / Output Example

启用CoT后,控制台会显示类似以下内容:

```
================================================================================
Chain-of-Thought (CoT) enabled with max_cot_tokens=100
================================================================================

================================================================================
[CoT][Step 0] Generalist Reasoning:
--------------------------------------------------------------------------------
First, I need to identify the current position of the gripper. Then, I should
determine the target object location. The red block appears to be at coordinates
(0.3, 0.2, 0.1). I need to move the gripper towards it while keeping the
gripper open. The movement should be smooth and gradual to avoid collision.
================================================================================

[Latency][System-2] Step 0: TTFT=0.1234s, TPOT=0.0156s, Tokens=125
```

## 验证 / Verification

### 验证Specialist不受CoT影响

Specialist的输入是从generalist的hidden states中提取的,具体逻辑在 `modeling_prismatic.py` 的585-599行:

```python
# Collect hidden states only for the last action_dim tokens
for i in range(max(0, num_generated - action_dim), num_generated):
    if i < len(hidden_states) and len(hidden_states[i]) > 0:
        last_layer_states.append(hidden_states[i][-1][:, -1:, :])
```

这确保了即使生成了CoT内容,specialist也只接收action tokens对应的hidden states。

### 测试建议

1. **对比实验**: 分别运行带CoT和不带CoT的评估,对比性能指标
2. **CoT质量检查**: 检查打印的CoT内容是否合理,是否包含有意义的推理步骤
3. **Specialist行为**: 验证specialist的行为在启用/禁用CoT时保持一致(因为它只接收action hidden states)

## 注意事项 / Notes

1. **推理延迟**: 启用CoT会增加generalist的推理时间,因为需要生成额外的tokens
2. **Token限制**: `max_cot_tokens` 设置应合理,太小可能截断推理,太大会增加延迟
3. **Zero-shot**: 当前实现使用zero-shot CoT,不需要额外训练数据
4. **模型能力**: CoT质量取决于底层LLM的推理能力

## 故障排除 / Troubleshooting

**问题1**: CoT内容为空或不合理
- **解决**: 增加 `max_cot_tokens` 值,给模型更多空间生成推理

**问题2**: 推理速度太慢
- **解决**: 减小 `max_cot_tokens` 值,或者禁用CoT

**问题3**: Tokenizer错误
- **解决**: 确保processor正确加载,并且包含有效的tokenizer

## 未来改进 / Future Improvements

1. **Few-shot CoT**: 提供示例推理过程
2. **CoT微调**: 使用标注的推理数据微调模型
3. **推理质量评估**: 添加自动评估CoT质量的指标
4. **自适应token分配**: 根据任务复杂度动态调整CoT长度

## 作者 / Author

实现日期: 2025-11-05
