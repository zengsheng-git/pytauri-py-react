#!/usr/bin/env python
"""简化测试训练功能 - 不依赖 pytauri"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pysrc'))

import threading
import queue
import time
from types import SimpleNamespace

print("=== 简化测试训练功能 ===")
print(f"Python 版本: {sys.version}")
print(f"当前目录: {os.getcwd()}")
print(f"sys.path: {sys.path[:3]}")

# 测试 mlx 导入
try:
    import mlx.core as mx
    import mlx.nn as nn
    print("✓ mlx 导入成功")
except ImportError as e:
    print(f"✗ mlx 导入失败: {e}")
    sys.exit(1)

# 测试 transformers 导入
try:
    from transformers import AutoTokenizer
    print("✓ transformers 导入成功")
except ImportError as e:
    print(f"✗ transformers 导入失败: {e}")
    sys.exit(1)

# 测试 mlx_lm_lora 导入
try:
    from mlx_lm_lora.train import train_model, from_pretrained, load_dataset
    from mlx_lm_lora.trainer.datasets import CacheDataset
    print("✓ mlx_lm_lora 导入成功")
except ImportError as e:
    print(f"✗ mlx_lm_lora 导入失败: {e}")
    # 尝试使用绝对路径导入
    try:
        from pysrc.mlx_lm_lora.train import train_model, from_pretrained, load_dataset
        from pysrc.mlx_lm_lora.trainer.datasets import CacheDataset
        print("✓ 使用绝对路径导入 mlx_lm_lora 成功")
    except ImportError as e2:
        print(f"✗ 绝对路径导入也失败: {e2}")
        sys.exit(1)

# 测试配置
config = SimpleNamespace(
    model="Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1",
    data="mlx-community/wikisql",
    train=True,
    train_type="lora",
    train_mode="sft",
    optimizer="adam",
    optimizer_config={"adam": {}, "adamw": {}, "muon": {}},
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
    lora_parameters={
        "rank": 8,
        "dropout": 0.0,
        "scale": 10.0
    },
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
    seed=0,
    resume_adapter_file=None,
    test=False,
    test_batches=500,
    lr_schedule=None,
    mask_prompt=False,
    fuse=True,
    reward_scaling=1.0,
    reward_weights=None,
    reward_functions=None,
    reward_functions_file=None,
    epsilon_high=None,
    importance_sampling_level=None,
    judge_config={},
    wandb=None,
)

print(f"\n配置: {config.model}")
print(f"训练模式: {config.train_mode}")
print(f"迭代次数: {config.iters}")

# 进度队列
progress_queue = queue.Queue()

def training_worker():
    try:
        print("\n启动训练线程...")
        progress_queue.put({"status": "starting", "message": "正在启动训练..."})
        
        # 加载模型
        print("正在加载模型...")
        progress_queue.put({"status": "loading_model", "message": "正在加载模型..."})
        
        model, tokenizer = from_pretrained(
            model=config.model, quantized_load=None
        )
        print(f"模型加载成功: {type(model)}")
        print(f"Tokenizer 加载成功: {type(tokenizer)}")
        
        # 加载数据集
        print("正在加载数据集...")
        progress_queue.put({"status": "loading_data", "message": "正在加载数据集..."})
        
        train_set, valid_set, test_set = load_dataset(config, tokenizer)
        print(f"数据集加载成功 - 训练集大小: {len(train_set)}")
        
        # 准备数据集
        train_dataset = CacheDataset(train_set)
        valid_dataset = CacheDataset(valid_set)
        
        # 开始训练
        print("正在开始训练...")
        progress_queue.put({"status": "training", "message": "正在开始训练..."})
        
        train_model(
            config,
            model,
            tokenizer,
            None,  # reference_model
            None,  # judge_model
            None,  # judge_tokenizer
            train_dataset,
            valid_dataset,
            None,  # test_dataset
        )
        
        print("训练完成！")
        progress_queue.put({"status": "completed", "message": "训练完成！"})
        
    except Exception as e:
        import traceback
        error_msg = f"训练出错: {str(e)}\n{traceback.format_exc()}"
        print(f"训练线程出错: {error_msg}")
        progress_queue.put({"status": "error", "message": error_msg})

# 启动训练线程
training_thread = threading.Thread(target=training_worker, daemon=True)
training_thread.start()

# 监控进度
print("\n监控训练进度...")
start_time = time.time()
timeout = 60  # 60秒超时

while time.time() - start_time < timeout:
    if not training_thread.is_alive():
        print("训练线程已结束")
        break
    
    try:
        progress = progress_queue.get_nowait()
        print(f"[{progress['status']}] {progress['message']}")
    except queue.Empty:
        time.sleep(1)
        continue

# 检查最终状态
if training_thread.is_alive():
    print("训练仍在进行中...")
else:
    print("训练线程已结束")

print("\n=== 测试完成 ===")
