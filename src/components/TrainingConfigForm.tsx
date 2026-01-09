import { useState, useEffect, useRef } from "react";
import { pyInvoke } from "tauri-plugin-pytauri-api";
import * as echarts from 'echarts'; // 引入ECharts
import styles from './TrainingConfigForm.module.css'; // 导入 CSS module

interface TrainingConfig {
  model: string;
  data: string;
  trainType: "lora" | "dora" | "full";
  trainMode: "sft" | "dpo" | "cpo" | "orpo" | "grpo" | "online_dpo" | "xpo" | "rlhf_reinforce" | "ppo";
  optimizer: "adam" | "adamw" | "muon";
  numLayers: number;
  batchSize: number;
  iters: number | null;
  epochs: number | null;
  gradientAccumulationSteps: number;
  valBatches: number;
  learningRate: number;
  stepsPerReport: number;
  stepsPerEval: number;
  adapterPath: string;
  saveEvery: number;
  maxSeqLength: number;
  gradCheckpoint: boolean;
  loraRank: number;
  loraDropout: number;
  loraScale: number;
  beta: number;
  lossType: "sigmoid" | "hinge" | "ipo" | "dpop";
  delta: number;
  referenceModelPath: string | null;
  judge: string | null;
  groupSize: number;
  maxCompletionLength: number;
  temperature: number;
  epsilon: number;
  alpha: number;
  grpoLossType: "grpo" | "bnpo" | "dr_grpo";
}

interface TrainingProgress {
  iteration: number;
  trainLoss: number | null;
  valLoss: number | null;
  learningRate: number | null;
  tokensPerSecond: number | null;
  iterationsPerSecond: number | null;
  peakMemory: number | null;
  status: string;
  message: string;
}

export default function TrainingConfigForm() {
  const [config, setConfig] = useState<TrainingConfig>({
    model: "Goekdeniz-Guelmez/Josiefied-Qwen2.5-0.5B-Instruct-abliterated-v1",
    data: "mlx-community/wikisql",
    trainType: "lora",
    trainMode: "sft",
    optimizer: "adam",
    numLayers: 16,
    batchSize: 4,
    iters: 600,
    epochs: null,
    gradientAccumulationSteps: 1,
    valBatches: 25,
    learningRate: 1e-5,
    stepsPerReport: 10,
    stepsPerEval: 200,
    adapterPath: "adapters",
    saveEvery: 100,
    maxSeqLength: 2048,
    gradCheckpoint: false,
    loraRank: 8,
    loraDropout: 0.0,
    loraScale: 10.0,
    beta: 0.1,
    lossType: "sigmoid",
    delta: 50.0,
    referenceModelPath: null,
    judge: null,
    groupSize: 4,
    maxCompletionLength: 512,
    temperature: 0.8,
    epsilon: 1e-4,
    alpha: 1e-5,
    grpoLossType: "grpo",
  });

  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState<TrainingProgress>({
    iteration: 0,
    trainLoss: null,
    valLoss: null,
    learningRate: null,
    tokensPerSecond: null,
    iterationsPerSecond: null,
    peakMemory: null,
    status: "idle",
    message: "",
  });
  const [message, setMessage] = useState("");
  const [logs, setLogs] = useState<string[]>([]);

  const [Xdata,setXdata] = useState([]);
  const [lossVal,setLossVal] = useState([]);
 // 1. 创建ref关联DOM容器
  const chartRef = useRef(null);
  // 2. 声明echarts实例变量（避免重复创建）
  const chartInstanceRef = useRef(null);

// 提取loss值和时间的函数
function extractLogInfo(logString: string) {
    // 定义正则：同时匹配时间和loss值（用两个捕获组）
    // 时间匹配：\[(\d{2}:\d{2})<  → 匹配[后、<前的 两位数字:两位数字
    // loss匹配：loss=(\d+\.\d+)  → 匹配loss=后的小数
    const logRegex = /\[(\d{2}:\d{2})<.*?loss=(\d+\.\d+)/;
    const matchResult = logRegex.exec(logString);

    // 初始化返回结果
    const result = {
        time: null,
        loss: null
    };

    // 处理匹配结果
    if (matchResult) {
        result.time = matchResult[1]; // 第一个捕获组：时间（如02:12）
        result.loss = Number(matchResult[2]); // 第二个捕获组：loss值（转数字）
    } else {
        console.warn("未找到时间或loss值");
    }

    return result;
}

  useEffect(() => {
    let intervalId: number;

    const checkStatus = async () => {
      try {
        const trainingStatus = await pyInvoke<boolean>("is_training");
        setIsTraining(trainingStatus);

        if (trainingStatus) {
          const progressData = await pyInvoke<TrainingProgress>("get_training_progress");
          setProgress(progressData);
          console.log(progressData);
         
          const logInfo = extractLogInfo(progressData.message);
          if (logInfo.loss !== null) {
            setXdata((prev) => [...prev, logInfo.time]);
            setLossVal((prev) => [...prev, logInfo.loss]);
          }
          
          
          const allLogs = await pyInvoke<string[]>("get_all_logs");
          if (allLogs.length > 0) {
            setLogs((prev) => [...prev, ...allLogs]);
          }
        }
      } catch (error) {
        console.error("Error checking training status:", error);
      }
    };

    checkStatus();
    intervalId = window.setInterval(checkStatus, 1000);

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, []);

useEffect(() => {
    // 3. 初始化ECharts实例
    if (chartRef.current) {
      const chartInstance = echarts.init(chartRef.current);
      chartInstanceRef.current = chartInstance;
    }

    // 6. 组件卸载时销毁实例，防止内存泄漏
    return () => {
      if (chartInstanceRef.current) {
        chartInstanceRef.current.dispose();
        chartInstanceRef.current = null;
      }
    };
  }, []); // 空依赖数组：仅在组件挂载/卸载时执行

  // Update chart when loss history changes
  useEffect(() => {
    if (chartInstanceRef.current) {
      chartInstanceRef.current.setOption({
         title: {
          text: 'Loss Curve'
        },
        tooltip: {
          trigger: 'axis'
        },
        grid: {
          left: '3%',
          right: '4%',
          bottom: '3%',
          containLabel: true
        },
        
        xAxis: {
          type: 'category',
          data: Xdata
        },
        yAxis: {
          type: 'value',
          name: 'Loss',
          axisLabel: {
            formatter: '{value}'
          }
        },
        series: [
          {
            name: 'Validation Loss',
            type: 'line',
            data: lossVal,
            smooth: true,
            showSymbol: false
          }
        ],
      });
    }
    // console.log(lossVal);
    // console.log(Xdata);
  }, [lossVal]);

  const handleStartTraining = async () => {
    try {
      console.log("开始训练，配置:", config);
      setLogs([]);
      // 重置 loss 历史记录
      setXdata([]);
      setLossVal([]);
      const result = await pyInvoke<string>("start_training", config);
      console.log("训练启动结果:", result);
      setMessage(result);
      setIsTraining(true);
    } catch (error) {
      console.error("启动训练失败:", error);
      setMessage(`启动训练失败: ${error}`);
    }
  };

  const handleStopTraining = async () => {
    try {
      const result = await pyInvoke<string>("stop_training");
      setMessage(result);
    } catch (error) {
      setMessage(`停止训练失败: ${error}`);
    }
  };

  const handleInputChange = (field: keyof TrainingConfig, value: any) => {
    setConfig((prev) => ({ ...prev, [field]: value }));
  };

  return (
    <div className={styles.trainingForm}>
      <h1>MLX-LM-LoRA 训练配置</h1>

      <div className={styles.modulesContainer}>
        <div className={styles.formSection}>
          <h2>基本配置</h2>
          <div className={styles.formRow}>
            <label>
              model 模型路径:
              <input
                type="text"
                value={config.model}
                onChange={(e) => handleInputChange("model", e.target.value)}
                placeholder="例如: mlx-community/Qwen2.5-7B-Instruct-4bit"
              />
            </label>
            <label>
              data 数据集路径:
              <input
                type="text"
                value={config.data}
                onChange={(e) => handleInputChange("data", e.target.value)}
                placeholder="例如: ./data/train.jsonl"
              />
            </label>
            <label>
              trainType 训练类型:
              <select
                value={config.trainType}
                onChange={(e) => handleInputChange("trainType", e.target.value)}
              >
                <option value="lora">LoRA</option>
                <option value="dora">DoRA</option>
                <option value="full">Full</option>
              </select>
            </label>
            <label>
              trainMode 训练模式:
              <select
                value={config.trainMode}
                onChange={(e) => handleInputChange("trainMode", e.target.value)}
              >
                <option value="sft">SFT</option>
                <option value="dpo">DPO</option>
                <option value="cpo">CPO</option>
                <option value="orpo">ORPO</option>
                <option value="grpo">GRPO</option>
                <option value="online_dpo">Online DPO</option>
                <option value="xpo">XPO</option>
                <option value="rlhf_reinforce">RLHF Reinforce</option>
                <option value="ppo">PPO</option>
              </select>
            </label>
          </div>
        </div>

        <div className={styles.formSection}>
          <h2>训练参数</h2>
          <div className={styles.formRow}>
            <label>
              batchSize 批次大小:
              <input
                type="number"
                value={config.batchSize}
                onChange={(e) => handleInputChange("batchSize", parseInt(e.target.value))}
              />
            </label>
            <label>
              learningRate 学习率:
              <input
                type="number"
                step="1e-6"
                value={config.learningRate}
                onChange={(e) => handleInputChange("learningRate", parseFloat(e.target.value))}
              />
            </label>
            <label>
              iters 迭代次数:
              <input
                type="number"
                value={config.iters || ""}
                onChange={(e) => handleInputChange("iters", e.target.value ? parseInt(e.target.value) : null)}
                placeholder="留空则使用 epochs"
              />
            </label>
            <label>
              epochs 训练轮数:
              <input
                type="number"
                value={config.epochs || ""}
                onChange={(e) => handleInputChange("epochs", e.target.value ? parseInt(e.target.value) : null)}
                placeholder="留空则使用 iters"
              />
            </label>
            <label>
              maxSeqLength 最大序列长度:
              <input
                type="number"
                value={config.maxSeqLength}
                onChange={(e) => handleInputChange("maxSeqLength", parseInt(e.target.value))}
              />
            </label>
          </div>
        </div>

        <div className={styles.formSection}>
          <h2>LoRA 参数</h2>
          <div className={styles.formRow}>
            <label>
              loraRank LoRA 秩:
              <input
                type="number"
                value={config.loraRank}
                onChange={(e) => handleInputChange("loraRank", parseInt(e.target.value))}
              />
            </label>
            <label>
              LoRA Dropout:
              <input
                type="number"
                step="0.01"
                value={config.loraDropout}
                onChange={(e) => handleInputChange("loraDropout", parseFloat(e.target.value))}
              />
            </label>
            <label>
              LoRA Scale:
              <input
                type="number"
                step="0.1"
                value={config.loraScale}
                onChange={(e) => handleInputChange("loraScale", parseFloat(e.target.value))}
              />
            </label>
          </div>
        </div>

        <div className={styles.formSection}>
          <h2>高级参数</h2>
          <div className={styles.formRow}>
            <label>
              optimizer 优化器:
              <select
                value={config.optimizer}
                onChange={(e) => handleInputChange("optimizer", e.target.value)}
              >
                <option value="adam">Adam</option>
                <option value="adamw">AdamW</option>
                <option value="muon">Muon</option>
              </select>
            </label>
            <label>
               梯度累积步数:
              <input
                type="number"
                value={config.gradientAccumulationSteps}
                onChange={(e) => handleInputChange("gradientAccumulationSteps", parseInt(e.target.value))}
              />
            </label>
            <label>
              valBatches 验证批次数量:
              <input
                type="number"
                value={config.valBatches}
                onChange={(e) => handleInputChange("valBatches", parseInt(e.target.value))}
              />
            </label>
            <label>
               每多少步报告一次:
              <input
                type="number"
                value={config.stepsPerReport}
                onChange={(e) => handleInputChange("stepsPerReport", parseInt(e.target.value))}
              />
            </label>
            <label>
              stepsPerEval 每多少步验证一次:
              <input
                type="number"
                value={config.stepsPerEval}
                onChange={(e) => handleInputChange("stepsPerEval", parseInt(e.target.value))}
              />
            </label>
            <label>
              saveEvery 每多少步保存一次:
              <input
                type="number"
                value={config.saveEvery}
                onChange={(e) => handleInputChange("saveEvery", parseInt(e.target.value))}
              />
            </label>
            <label>
              adapterPath 适配器保存路径:
              <input
                type="text"
                value={config.adapterPath}
                onChange={(e) => handleInputChange("adapterPath", e.target.value)}
              />
            </label>
            <label>
              gradCheckpoint 使用梯度检查点:
              <input
                type="checkbox"
                checked={config.gradCheckpoint}
                onChange={(e) => handleInputChange("gradCheckpoint", e.target.checked)}
              />
            </label>
          </div>
        </div>

        <div className={styles.formSection}>
          <h2>DPO/ORPO 参数</h2>
          <div className={styles.formRow}>
            <label>
              Beta:
              <input
                type="number"
                step="0.01"
                value={config.beta}
                onChange={(e) => handleInputChange("beta", parseFloat(e.target.value))}
              />
            </label>
            <label>
              lossType 损失类型:
              <select
                value={config.lossType}
                onChange={(e) => handleInputChange("lossType", e.target.value)}
              >
                <option value="sigmoid">Sigmoid</option>
                <option value="hinge">Hinge</option>
                <option value="ipo">IPO</option>
                <option value="dpop">DPOP</option>
              </select>
            </label>
            <label>
              Delta:
              <input
                type="number"
                step="0.1"
                value={config.delta}
                onChange={(e) => handleInputChange("delta", parseFloat(e.target.value))}
              />
            </label>
          </div>
        </div>
      </div>

      <div className={styles.formSection}>
        <h2>训练控制</h2>
        <div className={styles.buttonGroup}>
          <button
            onClick={handleStartTraining}
            disabled={isTraining}
            className={styles.startButton}
          >
            开始训练
          </button>
          <button
            onClick={handleStopTraining}
            disabled={!isTraining}
            className={styles.stopButton}
          >
            停止训练
          </button>
        </div>
        {message && <div className={styles.message}>{message}</div>}
      </div>
      <div className={styles.lossChart} ref={chartRef} style={{ height: '400px' }}></div>

      {isTraining && (
        <div className={styles.progressSection}>
          <h2>训练进度</h2>
          <div className={styles.progressInfo}>
            <div><strong>状态:</strong> {progress.status}</div>
            <div><strong>消息:</strong> {progress.message}</div>
            {progress.iteration > 0 && <div><strong>迭代:</strong> {progress.iteration}</div>}
            {progress.trainLoss != null && <div><strong>训练损失:</strong> {progress.trainLoss.toFixed(4)}</div>}
            {progress.valLoss != null && <div><strong>验证损失:</strong> {progress.valLoss.toFixed(4)}</div>}
            {progress.learningRate != null && <div><strong>学习率:</strong> {progress.learningRate.toExponential(2)}</div>}
            {progress.iterationsPerSecond != null && <div><strong>迭代/秒:</strong> {progress.iterationsPerSecond.toFixed(2)}</div>}
            {progress.tokensPerSecond != null && <div><strong>Token/秒:</strong> {progress.tokensPerSecond.toFixed(2)}</div>}
            {progress.peakMemory != null && <div><strong>峰值内存:</strong> {progress.peakMemory.toFixed(2)} GB</div>}
          </div>
          {logs.length > 0 && (
            <div className={styles.progressLog}>
              <h3>训练日志</h3>
              <div className={styles.logContent}>
                {logs.map((log, index) => (
                  <div key={index} className={styles.logEntry}>{log}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
