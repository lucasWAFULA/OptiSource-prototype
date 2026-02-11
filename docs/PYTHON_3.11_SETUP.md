# Use Python 3.11 for ML-TSSP (Windows)

TensorFlow/Keras GRU requires Python 3.10–3.12. This guide sets **Python 3.11** as the one you use for this project.

---

## Option A: Project-only (recommended) – venv with 3.11

Use Python 3.11 only for this project; leave your system default as-is.

### 1. Install Python 3.11

- Download: https://www.python.org/downloads/release/python-3119/
- Run the **Windows installer (64-bit)**.
- On the first screen, check **"Add python.exe to PATH"**.
- Choose **"Customize installation"** and ensure **"py launcher for all users"** is checked → Next → Install.

### 2. Create a virtual environment with 3.11

Open **PowerShell** or **Command Prompt** and run:

```powershell
cd "D:\Updated-FINAL DASH"
py -3.11 -m venv .venv
```

If you see `'py' is not recognized`, use the full path to Python 3.11 instead, for example:

```powershell
& "C:\Users\hp\AppData\Local\Programs\Python\Python311\python.exe" -m venv .venv
```

(Adjust the path if your Python 3.11 is installed elsewhere.)

### 3. Activate the venv and install requirements

**PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Command Prompt:**

```cmd
.venv\Scripts\activate.bat
```

You should see `(.venv)` at the start of the line. Then:

```powershell
python --version
pip install -r requirements.txt
```

You should see **Python 3.11.x**. From now on, in this terminal (with the venv active), `python` and `pip` are 3.11.

### 4. Run the app

```powershell
streamlit run streamlit_app.py
```

To use this project later: open a terminal → `cd "D:\Updated-FINAL DASH"` → activate the venv → run the app. No need to change the system-wide default.

---

## Option B: System-wide default to Python 3.11

Make `python` and `pip` point to 3.11 everywhere (can affect other projects).

### 1. Install Python 3.11

Same as Option A, step 1. Ensure **"Add python.exe to PATH"** is checked.

### 2. Put Python 3.11 before 3.13 in PATH

1. Press **Win**, type **environment variables**, open **Edit the system environment variables**.
2. Click **Environment Variables**.
3. Under **User variables** (or **System variables**), select **Path** → **Edit**.
4. Find entries like:
   - `C:\Users\hp\AppData\Local\Programs\Python\Python311\`
   - `C:\Users\hp\AppData\Local\Programs\Python\Python311\Scripts\`
   - And similar for Python313 (or 3.13).
5. **Move the Python 3.11 entries above the Python 3.13 entries** (use **Move Up**).
6. Click **OK** on all dialogs.

### 3. Confirm in a new terminal

Close and reopen PowerShell/Command Prompt, then:

```powershell
python --version
```

You should see **Python 3.11.x**. Then:

```powershell
cd "D:\Updated-FINAL DASH"
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Quick check

- **Option A:** In the project folder, after activating `.venv`, run `python --version` → should be 3.11.x.
- **Option B:** In any new terminal, run `python --version` → should be 3.11.x.

If you still see 3.13, the PATH order is wrong (Option B) or the venv was created with another Python (Option A: recreate with `py -3.11 -m venv .venv`).
