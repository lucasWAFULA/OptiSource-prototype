# Step-by-Step: Link Supabase/PostgreSQL with ML-TSSP

Follow these steps in order to connect your ML-TSSP system to Supabase (PostgreSQL) so optimization results and assignments are shared across instances or regions.

---

## What you’ll need

- A Supabase account (free at [supabase.com](https://supabase.com))
- This repo on your machine (with `scripts/init_postgres_schema.sql` and the app code)
- Python environment with `pip install -r requirements.txt` already run

---

## Step 1: Create a Supabase account and project

1. Open a browser and go to **https://supabase.com**.
2. Click **Start your project** (or **Sign in** if you already have an account).
3. Sign in with GitHub or email and complete sign-up if needed.
4. After logging in, click **New project**.
5. Fill in:
   - **Organization**: Use the default or create one.
   - **Name**: e.g. `ml-tssp` or `optisource`.
   - **Database Password**: Create a **strong password** and **save it somewhere safe** (you’ll need it for the connection string).
   - **Region**: Choose the region closest to you (e.g. **East US (N. Virginia)** or **Europe (Frankfurt)**).
6. Click **Create new project**.
7. Wait 1–2 minutes until the project status is **Active** (green).

---

## Step 2: Get your PostgreSQL connection string

1. In the Supabase dashboard, in the **left sidebar**, click the **gear icon** (⚙️) at the bottom → **Project Settings**.
2. In the left menu under **Project Settings**, click **Database**.
3. Scroll to **Connection string**.
4. Open the **URI** tab (not “Session mode” or “Transaction” for now).
5. You’ll see a URI like:
   ```text
   postgresql://postgres.[project-ref]:[YOUR-PASSWORD]@aws-0-[region].pooler.supabase.com:6543/postgres
   ```
6. Click **Copy** (or select and copy the whole string).
7. **Replace `[YOUR-PASSWORD]`** in that string with the **database password** you set in Step 1.  
   Example:  
   - Before: `postgresql://postgres.abcdefgh:YOUR-PASSWORD@aws-0-us-east-1.pooler.supabase.com:6543/postgres`  
   - After:  `postgresql://postgres.abcdefgh:MyStr0ngPass123@aws-0-us-east-1.pooler.supabase.com:6543/postgres`
8. **Optional but recommended for Supabase:** Add SSL to the end of the URI so it ends with:
   ```text
   postgres?sslmode=require
   ```
   So the full URI looks like:
   ```text
   postgresql://postgres.xxxx:YourPassword@aws-0-xx.pooler.supabase.com:6543/postgres?sslmode=require
   ```
9. Save this final URI somewhere safe (e.g. a password manager). You’ll use it as `DATABASE_URL` in Step 5.

---

## Step 3: Create the database tables (schema) in Supabase

1. In the Supabase **left sidebar**, click **SQL Editor**.
2. Click **New query** (or the **+** to create a new query).
3. On your computer, open the file **`scripts/init_postgres_schema.sql`** from this repo (in the same folder as `dashboard.py`).
4. Select **all** the SQL in that file (Ctrl+A / Cmd+A) and **copy** it.
5. In the Supabase SQL Editor, **paste** the SQL into the query box (replace any existing text).
6. Click **Run** (or press Ctrl+Enter / Cmd+Enter).
7. At the bottom you should see a success message (e.g. “Success. No rows returned”).  
   This creates the tables ML-TSSP needs: `sources`, `assignments`, `optimization_results`, `audit_log`, etc.
8. **Verify:** In the left sidebar click **Table Editor**. You should see tables such as `sources`, `assignments`, `optimization_results`, `audit_log`. If you see them, the schema is ready.

---

## Step 4: Install Python dependencies (if not already done)

1. Open a terminal (or PowerShell / Command Prompt) on your computer.
2. Go to your project folder (where `requirements.txt` and `dashboard.py` are):
   ```bash
   cd "d:\Updated-FINAL DASH"
   ```
   (Use your actual path if different.)
3. Create/activate a virtual environment if you use one, then run:
   ```bash
   pip install -r requirements.txt
   ```
4. This installs `psycopg2-binary` (and everything else) needed for PostgreSQL. No need to install anything else for the database.

---

## Step 5: Set DATABASE_URL and run ML-TSSP

You must set the environment variable `DATABASE_URL` to your Supabase URI **before** starting the app.

### Option A – Set in the terminal (good for testing)

**Windows (Command Prompt):**
```cmd
set DATABASE_URL=postgresql://postgres.YOUR_REF:YOUR_PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres?sslmode=require
streamlit run streamlit_app.py
```

**Windows (PowerShell):**
```powershell
$env:DATABASE_URL="postgresql://postgres.YOUR_REF:YOUR_PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres?sslmode=require"
streamlit run streamlit_app.py
```

**Linux / macOS:**
```bash
export DATABASE_URL="postgresql://postgres.YOUR_REF:YOUR_PASSWORD@aws-0-REGION.pooler.supabase.com:6543/postgres?sslmode=require"
streamlit run streamlit_app.py
```

Replace `YOUR_REF`, `YOUR_PASSWORD`, and `REGION` with the values from your Supabase URI (from Step 2). Use the **exact** URI you saved, including `?sslmode=require` if you added it.

### Option B – Use a `.env` file (good for keeping secrets out of the command line)

1. In your project folder (same place as `dashboard.py`), create a file named **`.env`**.
2. Add one line (use your real URI from Step 2):
   ```env
   DATABASE_URL=postgresql://postgres.xxxx:YourPassword@aws-0-xx.pooler.supabase.com:6543/postgres?sslmode=require
   ```
3. Save the file.
4. **Important:** The app must **load** this file. If you use `python-dotenv`, add at the very top of `streamlit_app.py` or `dashboard.py` (before any other imports that use the DB):
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```
   Then install it if needed: `pip install python-dotenv`
5. Start the app **without** setting `DATABASE_URL` in the terminal:
   ```bash
   streamlit run streamlit_app.py
   ```

---

## Step 6: Confirm ML-TSSP is using Supabase

1. With the app running, open the dashboard in your browser (e.g. `http://localhost:8501`).
2. Log in if your app has login.
3. Find the **health** or **system status** section (often in the sidebar or a “Health” page).
4. Check the **shared database** line. It should say something like:
   - **Engine:** `postgres`
   - **Mode:** `PostgreSQL/Supabase`
   - **Connected:** `True`
5. If it still says **Local (SQLite)** or **Not connected**:
   - Stop the app (Ctrl+C in the terminal).
   - Confirm `DATABASE_URL` is set in the **same** terminal you use to run `streamlit run ...` (Option A), or that `.env` is loaded (Option B).
   - Start the app again and recheck.

---

## Step 7: Verify shared results (optional but recommended)

1. **First instance:** With `DATABASE_URL` set, run the dashboard. Upload or generate data, then **Run optimization**. Note the results (e.g. assignments or EMV).
2. **Second instance:** Open a **second** terminal, set the **same** `DATABASE_URL` (same URI), and run the dashboard again (e.g. on port 8502: `streamlit run streamlit_app.py --server.port 8502`). Or use another computer with the same `DATABASE_URL`.
3. In the second instance, open the view that shows **shared** or **latest** optimization results (or refresh). You should see the **same** results as in the first instance.
4. Run an optimization from the second instance; then in the first instance refresh or reload shared results. You should see the **updated** data.

If both instances show the same data, ML-TSSP is successfully linked to Supabase/PostgreSQL and results are shared.

---

## Summary checklist

| Step | Action |
|------|--------|
| 1 | Create Supabase account and **New project**; save **database password**. |
| 2 | **Project Settings** → **Database** → copy **URI**; replace `[YOUR-PASSWORD]`; add `?sslmode=require`; save as your connection string. |
| 3 | **SQL Editor** → **New query** → paste **`scripts/init_postgres_schema.sql`** → **Run**; confirm tables in **Table Editor**. |
| 4 | In project folder: `pip install -r requirements.txt`. |
| 5 | Set **`DATABASE_URL`** (terminal or `.env`) to your Supabase URI, then run `streamlit run streamlit_app.py`. |
| 6 | In the app, check health: **Engine** = `postgres`, **Mode** = `PostgreSQL/Supabase`. |
| 7 | (Optional) Run two instances with same `DATABASE_URL` and confirm they see the same optimization results. |

---

## Troubleshooting

| Problem | What to do |
|--------|------------|
| **Connection refused / timeout** | Check the URI (host, port 6543, password). Ensure Supabase project is **Active**. Allow outbound HTTPS/Postgres from your network/firewall. |
| **SSL error** | Make sure the URI ends with `?sslmode=require` (or that the Supabase UI shows SSL in the connection string). |
| **“Still using SQLite”** | `DATABASE_URL` must be set in the **same** environment that starts the app. In the terminal, run `echo %DATABASE_URL%` (Windows) or `echo $DATABASE_URL` (Linux/macOS) before starting Streamlit; it should print your URI. |
| **Tables already exist** | The script uses `CREATE TABLE IF NOT EXISTS`. Safe to run again; it won’t overwrite data. |
| **Password has special characters** | If your DB password contains `@`, `#`, `%`, etc., URL-encode that character in the URI (e.g. `%40` for `@`) or use a password without special characters for the database user. |
| **pgAdmin: “failed to resolve host” (getaddrinfo failed)** | Use the **pooler** host instead of the direct host: in Supabase go to **Project Settings → Database**, choose the **Session** or **Transaction** connection string and use that **host** (e.g. `aws-0-us-east-1.pooler.supabase.com`) and **port 6543**. In pgAdmin set **Host** to that pooler host, **Port** to `6543`, **Username** to `postgres.YOUR_PROJECT_REF` (e.g. `postgres.yjtcaafvhlnneixtjuid`), and enable SSL. Also check that the project is not **paused** (restore it in the Supabase dashboard if needed). |

---

## What gets stored in Supabase

When linked, ML-TSSP uses these Supabase (PostgreSQL) tables for shared state:

- **optimization_results** – Full optimization output (policies, EMV, `ml_tssp` results).
- **assignments** – Per-source task assignments (e.g. policy type `ml_tssp`).
- **sources** – Source features and recourse rules.
- **audit_log** – Audit trail of actions.
- **threshold_settings** / **threshold_requests** – Thresholds and approvals.
- **tasking_requests** / **recommendations** – Workflow data.

All instances that use the same `DATABASE_URL` read and write these tables, so results are shared across regions or machines.
