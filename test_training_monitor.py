#!/usr/bin/env python
"""测试训练启动并监控进度"""
import sys
import time
sys.path.insert(0, 'pysrc')

from commands.training import TrainingConfig, _training_manager

print("=== 测试训练启动和监控 ===")

config = TrainingConfig(
    model="Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1",
    data="mlx-community/wikisql",
    train_type="lora",
    train_mode="sft",
    optimizer="adam",
    num_layers=16,
    batch_size=4,
    iters=10,
    epochs=None,
    gradient_accumulation_steps=1,
    val_batches=25,
    learning_rate=1e-5,
    steps_per_report=10,
    steps_per_eval=200,
    adapter_path="adapters",
    save_every=100,
    max_seq_length=2048,
    grad_checkpoint=False,
    lora_rank=8,
    lora_dropout=0.0,
    lora_scale=10.0,
    beta=0.1,
    loss_type="sigmoid",
    delta=50.0,
    reference_model_path=None,
    judge=None,
    group_size=4,
    max_completion_length=512,
    temperature=0.8,
    epsilon=1e-4,
    alpha=1e-5,
    grpo_loss_type="grpo",
)

print("\n启动训练...")
result = _training_manager.start_training(config, None)
print(f"启动结果: {result}")

print(f"\n训练状态: {_training_manager.is_training()}")
print(f"线程是否存活: {_training_manager._training_thread.is_alive() if _training_manager._training_thread else None}")

print("\n监控训练进度 (30秒)...")
for i in range(30):
    time.sleep(1)
    progress = _training_manager.get_progress()
    print(f"[{i+1}s] 状态: {progress.status}, 消息: {progress.message[:100] if progress.message else '无'}")
    
    if progress.status == "completed":
        print("训练完成！")
        break
    elif progress.status == "error":
        print(f"训练出错: {progress.message}")
        break

print(f"\n最终训练状态: {_training_manager.is_training()}")
print(f"最终线程状态: {_training_manager._training_thread.is_alive() if _training_manager._training_thread else None}")

print("\n=== 测试完成 ===")
