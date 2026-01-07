#!/usr/bin/env python
"""测试训练启动"""
import sys
sys.path.insert(0, 'pysrc')

from commands.training import TrainingConfig, _training_manager
import time

print("=== 测试训练启动 ===")

# 创建测试配置
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

print(f"配置: {config.model}")
print(f"训练模式: {config.train_mode}")
print(f"迭代次数: {config.iters}")

# 启动训练
print("\n启动训练...")
result = _training_manager.start_training(config, None)
print(f"启动结果: {result}")

# 等待一下
time.sleep(2)

# 检查状态
print(f"\n训练状态: {_training_manager.is_training()}")
print(f"线程是否存活: {_training_manager._training_thread.is_alive() if _training_manager._training_thread else None}")

# 检查队列
print(f"\n队列大小: {_training_manager._progress_queue.qsize()}")

# 获取消息
print("\n队列中的消息:")
import queue
while True:
    try:
        msg = _training_manager._progress_queue.get_nowait()
        print(f"  [{msg.status}] {msg.message}")
    except queue.Empty:
        break

# 等待更多消息
print("\n等待 5 秒获取更多消息...")
time.sleep(5)

print("\n更多消息:")
while True:
    try:
        msg = _training_manager._progress_queue.get_nowait()
        print(f"  [{msg.status}] {msg.message}")
    except queue.Empty:
        break

print("\n=== 测试完成 ===")
