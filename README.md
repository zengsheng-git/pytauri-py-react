# pytauri-react-starter

Cross-platform desktop application starter built with Pytauri, React.js (TypeScript), and Python Wheel — combining modern web UIs with powerful Python backends.

---

## 🛠️ Project Initialization Guide

Follow these steps to set up the project after cloning the repository.

---

### 1. Clone the Repository

```bash
git clone https://github.com/hayatullah87/pytauri-react-starter.git
cd pytauri-react-starter
```

### 2. Set Up Python Environment (Backend)

Make sure you have [uv](https://github.com/astral-sh/uv) installed.

If not, install it:

```bash
pip install uv
```

Then create and activate a virtual environment:

```bash
uv venv .venv
```

Activate it:

* **macOS/Linux** :
  ```bash
  source .venv/bin/activate
  ```
* **Windows (PowerShell)** :
  ```bash
  .venv\Scripts\Activate.ps1
  ```

### 3. Install Python Dependencies

Install all project dependencies declared in `pyproject.toml`:

```bash
uv sync
```

### 4. Install Frontend dependencies

Install all frontend dependencies declared in `package.json`:

```bash
yarn install
```

### 5. Run Application

**For Development Mode (with hot reload):**
1. First, start the frontend development server:
   ```bash
   yarn dev
   ```
2. Then, in a separate terminal, run the Python app with development mode:

   **Windows PowerShell:**
   ```powershell
   $env:PYTAURI_REACT_DEV="1"; python main.py
   ```

   **Windows Command Prompt:**
   ```cmd
   set PYTAURI_REACT_DEV=1 && python main.py
   ```

   **macOS/Linux:**
   ```bash
   PYTAURI_REACT_DEV=1 python main.py
   ```

**For Production Mode:**
1. Build the frontend:
   ```bash
   yarn build
   ```
2. Run the Python app:
   ```bash
   python main.py
   ```

---

## 📦 Building Wheel Distribution

To create a Python wheel distribution of your app:

### Build Steps
1. **Build the frontend:**
   ```bash
   yarn build
   ```
2. **Build the wheel:**
   ```bash
   uv build
   ```
3. **Installation to Test the wheel:**
   ```bash
   pip install dist/pytauri_react_starter-0.1.0-py3-none-any.whl
   ```
3. ** Launch to Test the wheel:**
   ```bash
   pytauri-react-starter
   ```
5. **Publish to PyPI (optional):**
   ```bash
   uv publish
   ```

The wheel will be created in the `dist/` directory and can be distributed or uploaded to PyPI.

---

# Acknowledgement
This is a reimplementation of [pytauri-next-starter](https://github.com/JeenLabs/pytauri-next-starter), replacing Next.js with React and Vite, and adding ESLint support for improved code quality and consistency.
