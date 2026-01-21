source /Users/apple/Desktop/mlx/pytauri-py/reactpy.venv/bin/activate
cd /Users/apple/Desktop/mlx/pytauri-py && PYTAURI_REACT_DEV=1 python main.py
cd /Users/apple/Desktop/mlx/pytauri-py && npm run build


uv venv reactpy.venv && source reactpy.venv/bin/activate && uv pip install "mlx>=0.29.3" "transformers>=4.39.3" numpy datasets mlx-lm pytauri pytauri-wheel pytauri-plugins

uv venv reactpy.venv && source reactpy.venv/bin/activate && uv pip install "mlx>=0.29.3" "transformers>=4.39.3" numpy datasets mlx-lm pytauri pytauri-wheel


   ```bash
   uv venv reactpy.venv
   source ./reactpy.venv/bin/activate
   ```
   ```bash
   uv sync --active
   ```
   uv pip install "mlx>=0.29.3" "transformers>=4.39.3" numpy datasets mlx-lm pytauri pytauri-wheel

试试训练的其他命令是否有用

`/Users/apple/Desktop/mlx/pytauri-py-react/pysrc/mlx_lm_lora` 这个里面所有的训练参数都在前端页面的表单中吗？

把 mlx_lm_lora整合进去，需要跑起来。

难点 编写 training.py 让 前端去调用

难点 通读收集 所有的参数 去编写表单

难点 但是 react-desktop 没有表单组件比如下拉框，很多其他的都没有

难点 训练进度的实时体现

生成的东西放哪需要自定义

react-desktop 不支持 raect19 ，最高支持 react 16

打包成一个exe ，dmg 安装包

/Users/apple/Desktop/mlx/2.pdf

{
    "iteration": 0,
    "train_loss": null,
    "val_loss": null,
    "learning_rate": null,
    "tokens_per_second": null,
    "iterations_per_second": null,
    "peak_memory": null,
    "status": "training",
    "message": "Training:  46%|####6     | 277/600 [01:50<01:59,  2.71it/s, loss=0.948, it/s=2.849]"
}
[2.137,2.137,2.137,2.137,1.602,1.602,1.602,1.602,1.309,1.309,1.309,1.123,1.123]
["00:08","00:09","00:10","00:12","00:12","00:13","00:14","00:15","00:16","00:17","00:18","00:19","00:20"]

## 工作空间

### Embedding 训练数据生成
Input 目录 
Output 缓存目录
Output 目录
训练结果名称

### 模型服务配置
模型服务类型
模型名称
云端地址
API Key

### 配置管理
配置名称
`保存配置`
`读取已有配置`
`执行生成 Embedding 数据`



`/Users/apple/Desktop/mlx/pytauri-py-react` `/Users/apple/Desktop/mlx/mlx-lm-lora` 回顾之前我们的对话，说明这两个项目有什么关系

<!-- 

npm run build
npx tauri build --config src-tauri/tauri.conf.json
cd src-tauri && cargo build --release
cd src-tauri && npx tauri build

python3 -m pip install --user --break-system-packages -e .
python3 -m pip install --user --break-system-packages docling
python3 -m pip install --user --break-system-packages markdown2
python3 -m pip install --user --break-system-packages weasyprint

python3 -m pip install --user --break-system-packages hatchling
python3 -m pip install --user --break-system-packages hatch
python3 -m pip install --user --break-system-packages build
python3 -m hatch build
python3 -m build 

-->

python3 -m pip wheel --no-deps -w dist .
python3 -m pip install --user --break-system-packages dist/pytauri_react_starter-0.1.0-py3-none-any.whl
/Users/apple/Library/Python/3.13/bin/pytauri-react-starter

 which pytauri-react-starter
/Users/apple/Desktop/mlx/pytauri-py-react/reactpy.venv/bin/pytauri-react-starter

python3 -m pip install --user --break-system-packages pyinstaller

python3 -m pip show pytauri_wheel
python3 -m pip wheel .
python3 -m pip install --user --break-system-packages pytauri_react_starter-0.1.0-py3-none-any.whl