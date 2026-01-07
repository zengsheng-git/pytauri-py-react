from . import commands
from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pytauri.webview import WebviewWindow
import threading
import queue
from typing import Optional, Literal
import sys
import io


class _BaseModel(BaseModel):
    """Base model that accepts camelCase from JS and snake_case from Python."""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class TrainingConfig(_BaseModel):
    """Training configuration model."""
    model: str = Field(description="模型路径或 Hugging Face 仓库")
    data: str = Field(description="数据集路径或 Hugging Face 数据集名称")
    train_type: Literal["lora", "dora", "full"] = Field(default="lora", description="训练类型")
    train_mode: Literal["sft", "dpo", "cpo", "orpo", "grpo", "online_dpo", "xpo", "rlhf_reinforce", "ppo"] = Field(default="sft", description="训练模式")
    optimizer: Literal["adam", "adamw", "muon"] = Field(default="adam", description="优化器")
    num_layers: int = Field(default=16, description="要微调的层数，-1 表示全部")
    batch_size: int = Field(default=4, description="批次大小")
    iters: Optional[int] = Field(default=None, description="训练迭代次数")
    epochs: Optional[int] = Field(default=None, description="训练轮数")
    gradient_accumulation_steps: int = Field(default=1, description="梯度累积步数")
    val_batches: int = Field(default=25, description="验证批次数量")
    learning_rate: float = Field(default=1e-5, description="学习率")
    steps_per_report: int = Field(default=10, description="每多少步报告一次损失")
    steps_per_eval: int = Field(default=200, description="每多少步验证一次")
    adapter_path: str = Field(default="adapters", description="适配器保存路径")
    save_every: int = Field(default=100, description="每多少步保存一次")
    max_seq_length: int = Field(default=2048, description="最大序列长度")
    grad_checkpoint: bool = Field(default=False, description="是否使用梯度检查点")
    lora_rank: int = Field(default=8, description="LoRA 秩")
    lora_dropout: float = Field(default=0.0, description="LoRA Dropout")
    lora_scale: float = Field(default=10.0, description="LoRA Scale")
    beta: float = Field(default=0.1, description="Beta 参数（用于 DPO/ORPO）")
    loss_type: Literal["sigmoid", "hinge", "ipo", "dpop"] = Field(default="sigmoid", description="损失类型")
    delta: float = Field(default=50.0, description="Delta 参数")
    reference_model_path: Optional[str] = Field(default=None, description="参考模型路径")
    judge: Optional[str] = Field(default=None, description="评估模型")
    group_size: int = Field(default=4, description="组大小")
    max_completion_length: int = Field(default=512, description="最大完成长度")
    temperature: float = Field(default=0.8, description="温度参数")
    epsilon: float = Field(default=1e-4, description="Epsilon 参数")
    alpha: float = Field(default=1e-5, description="Alpha 参数")
    grpo_loss_type: Literal["grpo", "bnpo", "dr_grpo"] = Field(default="grpo", description="GRPO 损失类型")


class TrainingProgress(_BaseModel):
    """Training progress model."""
    iteration: int = Field(default=0, description="当前迭代次数")
    train_loss: Optional[float] = Field(default=None, description="训练损失")
    val_loss: Optional[float] = Field(default=None, description="验证损失")
    learning_rate: Optional[float] = Field(default=None, description="学习率")
    tokens_per_second: Optional[float] = Field(default=None, description="每秒处理的 token 数")
    iterations_per_second: Optional[float] = Field(default=None, description="每秒迭代次数")
    peak_memory: Optional[float] = Field(default=None, description="峰值内存使用 (GB)")
    status: str = Field(default="idle", description="训练状态")
    message: str = Field(default="", description="状态消息")


class TrainingManager:
    """管理训练进程的单例类"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._training_thread = None
                    cls._instance._stop_event = threading.Event()
                    cls._instance._progress_queue = queue.Queue()
                    cls._instance._current_config = None
        return cls._instance
    
    def is_training(self) -> bool:
        """检查是否正在训练"""
        return self._training_thread is not None and self._training_thread.is_alive()
    
    def start_training(self, config: TrainingConfig, webview_window: WebviewWindow) -> str:
        """启动训练"""
        if self.is_training():
            return "训练已经在进行中"
        
        try:
            self._current_config = config
            self._stop_event.clear()
            
            def training_worker():
                try:
                    import sys
                    self._progress_queue.put(TrainingProgress(
                        iteration=0,
                        status="starting",
                        message="正在启动训练线程..."
                    ))
                    
                    from pysrc.mlx_lm_lora.train import train_model, from_pretrained, load_dataset, load_reference_model, load_judge_model
                    from pysrc.mlx_lm_lora.trainer.datasets import CacheDataset
                    from types import SimpleNamespace
                    import mlx.nn as nn
                    
                    args = SimpleNamespace(
                        model=config.model,
                        data=config.data,
                        train=True,
                        train_type=config.train_type,
                        train_mode=config.train_mode,
                        optimizer=config.optimizer,
                        optimizer_config={"adam": {}, "adamw": {}, "muon": {}},
                        num_layers=config.num_layers,
                        batch_size=config.batch_size,
                        iters=config.iters,
                        epochs=config.epochs,
                        gradient_accumulation_steps=config.gradient_accumulation_steps,
                        val_batches=config.val_batches,
                        learning_rate=config.learning_rate,
                        steps_per_report=config.steps_per_report,
                        steps_per_eval=config.steps_per_eval,
                        adapter_path=config.adapter_path,
                        save_every=config.save_every,
                        max_seq_length=config.max_seq_length,
                        grad_checkpoint=config.grad_checkpoint,
                        lora_parameters={
                            "rank": config.lora_rank,
                            "dropout": config.lora_dropout,
                            "scale": config.lora_scale
                        },
                        beta=config.beta,
                        loss_type=config.loss_type,
                        delta=config.delta,
                        reference_model_path=config.reference_model_path,
                        judge=config.judge,
                        group_size=config.group_size,
                        max_completion_length=config.max_completion_length,
                        temperature=config.temperature,
                        epsilon=config.epsilon,
                        alpha=config.alpha,
                        grpo_loss_type=config.grpo_loss_type,
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
                    
                    self._progress_queue.put(TrainingProgress(
                        iteration=0,
                        status="loading_model",
                        message="正在加载模型..."
                    ))
                    
                    quanziation_config = None
                    
                    model, tokenizer = from_pretrained(
                        model=args.model, quantized_load=quanziation_config
                    )
                    
                    reference_model = (
                        load_reference_model(args)
                        if args.train_mode in ["grpo", "online_dpo", "ppo", "rlhf_reinforce", "xpo"]
                        else None
                    )
                    judge_model, judge_tokenizer = (
                        load_judge_model(args, reference_model)
                        if args.train_mode in ["online_dpo", "ppo", "rlhf_reinforce", "xpo"]
                        else (None, None)
                    )
                    
                    self._progress_queue.put(TrainingProgress(
                        iteration=0,
                        status="loading_data",
                        message="正在加载数据集..."
                    ))
                    
                    train_set, valid_set, test_set = load_dataset(args, tokenizer)
                    
                    class LogCapture(io.StringIO):
                        def __init__(self, queue):
                            super().__init__()
                            self.queue = queue
                        
                        def write(self, text):
                            if text.strip():
                                self.queue.put(TrainingProgress(
                                    iteration=0,
                                    status="training",
                                    message=text.strip()
                                ))
                                super().write(text)
                        
                        def flush(self):
                            pass
                    
                    log_capture = LogCapture(self._progress_queue)
                    
                    old_stdout = sys.stdout
                    old_stderr = sys.stderr
                    sys.stdout = log_capture
                    sys.stderr = log_capture
                    
                    self._progress_queue.put(TrainingProgress(
                        iteration=0,
                        status="training",
                        message="正在开始训练..."
                    ))
                    
                    try:
                        train_model(
                            args,
                            model,
                            tokenizer,
                            reference_model,
                            judge_model,
                            judge_tokenizer,
                            CacheDataset(train_set),
                            CacheDataset(valid_set),
                            None,
                        )
                    except Exception as e:
                        import traceback
                        error_msg = f"训练出错: {str(e)}\n{traceback.format_exc()}"
                        self._progress_queue.put(TrainingProgress(
                            iteration=0,
                            status="error",
                            message=error_msg
                        ))
                        raise
                    finally:
                        sys.stdout = old_stdout
                        sys.stderr = old_stderr
                    
                    self._progress_queue.put(TrainingProgress(
                        iteration=config.iters or 0,
                        status="completed",
                        message="训练完成！"
                    ))
                    
                except Exception as e:
                    import traceback
                    error_msg = f"训练线程出错: {str(e)}\n{traceback.format_exc()}"
                    self._progress_queue.put(TrainingProgress(
                        iteration=0,
                        status="error",
                        message=error_msg
                    ))
            
            self._training_thread = threading.Thread(target=training_worker, daemon=True)
            self._training_thread.start()
            
            return "训练已启动"
        
        except Exception as e:
            import traceback
            error_msg = f"启动训练失败: {str(e)}\n{traceback.format_exc()}"
            return error_msg
    
    def stop_training(self) -> str:
        """停止训练"""
        if not self.is_training():
            return "没有正在进行的训练"
        
        self._stop_event.set()
        self._progress_queue.put(TrainingProgress(
            iteration=0,
            status="stopping",
            message="正在停止训练..."
        ))
        
        return "正在停止训练"
    
    def get_progress(self) -> TrainingProgress:
        """获取训练进度"""
        try:
            progress = self._progress_queue.get_nowait()
            return progress
        except queue.Empty:
            if self.is_training():
                return TrainingProgress(
                    status="running",
                    message=""
                )
            else:
                return TrainingProgress(
                    status="idle",
                    message=""
                )
    
    def get_all_logs(self) -> list[str]:
        """获取所有累积的日志"""
        logs = []
        while True:
            try:
                progress = self._progress_queue.get_nowait()
                if progress.message:
                    logs.append(progress.message)
            except queue.Empty:
                break
        return logs


_training_manager = TrainingManager()


@commands.command()
async def start_training(body: dict, webview_window: WebviewWindow) -> str:
    """启动训练任务"""
    config = TrainingConfig(**body)
    return _training_manager.start_training(config, webview_window)


@commands.command()
async def stop_training() -> str:
    """停止训练任务"""
    return _training_manager.stop_training()


@commands.command()
async def get_training_progress() -> TrainingProgress:
    """获取训练进度"""
    return _training_manager.get_progress()


@commands.command()
async def is_training() -> bool:
    """检查是否正在训练"""
    return _training_manager.is_training()


@commands.command()
async def get_all_logs() -> list[str]:
    """获取所有训练日志"""
    return _training_manager.get_all_logs()
