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

`/Users/apple/Desktop/mlx/pytauri-py-react/pysrc/mlx_lm_lora` 这个里面所有的训练参数都在前端页面的表单中吗？

把 mlx_lm_lora整合进去，需要跑起来。

难点 编写 training.py 让 前端去调用

难点 通读收集 所有的参数 去编写表单

难点 训练进度的实时体现
