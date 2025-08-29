
# --- Hugging Face Spaces hardening: cache/config to /tmp to avoid permission issues ---
import os, pathlib
for var, sub in {
    "MPLCONFIGDIR": "matplotlib",
    "XDG_CACHE_HOME": "xdg",
    "HF_HOME": "hf",
    "PIP_CACHE_DIR": "pip",
    "HUGGINGFACE_HUB_CACHE": "hfhub",
    "NUMBA_CACHE_DIR": "numba",
}.items():
    try:
        d = pathlib.Path("/tmp")/sub
        d.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(d)
    except Exception:
        pass

# Streamlit specific stability tweaks
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
\n
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date
try:\n    import plotly.express as px\nexcept Exception as e:\n    raise ImportError('Plotly is required. Add `plotly>=5.20` to requirements.txt in your Space. Original error: %s' % e)
from contextlib import contextmanager

# ============ App Config ============
st.set_page_config(
    page_title="Family Finance Manager",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ Constants ============
DB_PATH = "family_finance.db"
APP_USERS = ["George", "Jane"]
DEFAULT_CATEGORIES = {
    "Income": [
        "Salary", "Bonus", "Freelance", "Gift", "Other Income"
    ],
    "Expense": [
        "Rent/Mortgage", "Utilities", "Groceries", "Dining", "Transportation",
        "Insurance", "Healthcare", "Childcare", "Education", "Shopping",
        "Entertainment", "Travel", "Subscriptions", "Debt Payment", "Giving/Charity",
        "Investments", "Fees", "Other Expense"
    ]
}

# ============ DB Helpers ============
@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                ttype TEXT CHECK(ttype IN ('Income','Expense')) NOT NULL,
                category TEXT NOT NULL,
                amount REAL NOT NULL CHECK(amount >= 0),
                tdate TEXT NOT NULL,   -- ISO date
                description TEXT DEFAULT ''
            );
            '''
        )

def insert_transaction(user, ttype, category, amount, tdate, description):
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO transactions (user, ttype, category, amount, tdate, description) VALUES (?, ?, ?, ?, ?, ?);",
            (user, ttype, category, float(amount), tdate.isoformat(), description or "")
        )

def delete_transactions(ids):
    if not ids:
        return
    with get_conn() as conn:
        qmarks = ",".join("?" for _ in ids)
        conn.execute(f"DELETE FROM transactions WHERE id IN ({qmarks});", ids)

def fetch_transactions(user_filter=None, start=None, end=None):
    with get_conn() as conn:
        query = "SELECT id, user, ttype, category, amount, tdate, description FROM transactions WHERE 1=1"
        params = []
        if user_filter and user_filter != "All":
            query += " AND user = ?"
            params.append(user_filter)
        if start:
            query += " AND date(tdate) >= date(?)"
            params.append(start.isoformat())
        if end:
            query += " AND date(tdate) <= date(?)"
            params.append(end.isoformat())
        query += " ORDER BY date(tdate) DESC, id DESC;"
        df = pd.read_sql_query(query, conn, params=params)
    if not df.empty:
        # Ensure proper datetime dtype
        df["tdate"] = pd.to_datetime(df["tdate"], errors="coerce")
        df = df.dropna(subset=["tdate"])
        # Keep as date for pretty table display; we convert to datetime again where needed
        df["tdate"] = df["tdate"].dt.date
    return df

def summary_metrics(df):
    total_income = df.loc[df["ttype"] == "Income", "amount"].sum() if not df.empty else 0.0
    total_expense = df.loc[df["ttype"] == "Expense", "amount"].sum() if not df.empty else 0.0
    net = total_income - total_expense
    return total_income, total_expense, net

def monthly_rollup(df):
    # Robust guard and correct Series .dt usage
    if df.empty or "tdate" not in df.columns:
        return pd.DataFrame(columns=["month","ttype","amount"])
    temp = df.copy()
    temp["tdate"] = pd.to_datetime(temp["tdate"], errors="coerce")
    temp = temp.dropna(subset=["tdate"])
    # Correct: Series.dt.to_period(...).dt.to_timestamp()
    temp["month"] = temp["tdate"].dt.to_period("M").dt.to_timestamp()
    grp = temp.groupby(["month","ttype"], as_index=False)["amount"].sum()
    return grp

def category_rollup(df, ttype_filter="Expense", month=None):
    if df.empty:
        return pd.DataFrame(columns=["category","amount"])
    temp = df.copy()
    temp["tdate"] = pd.to_datetime(temp["tdate"], errors="coerce")
    temp = temp.dropna(subset=["tdate"])
    temp["month"] = temp["tdate"].dt.to_period("M").dt.to_timestamp()
    if month is not None:
        temp = temp[temp["month"] == month]
    if ttype_filter:
        temp = temp[temp["ttype"] == ttype_filter]
    grp = temp.groupby(["category"], as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    return grp

# ============ UI Helpers ============
def kpi_card(label, value):
    st.metric(label, f"${value:,.2f}")

def divider():
    st.markdown("<hr style='margin: 0.5rem 0;' />", unsafe_allow_html=True)

# ============ App ============
init_db()

# Sidebar: Identity & Filters
with st.sidebar:
    st.title("💸 Family Finance Manager")
    st.caption("Track income & expenses for George and Jane.")

    user = st.selectbox("Active user", APP_USERS, index=0, help="Entries you add will be recorded under this name.")
    st.divider()

    st.subheader("Filters")
    view_user = st.selectbox("View data for", ["All"] + APP_USERS, index=0)
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        start_date = st.date_input("Start date", value=date(date.today().year, 1, 1))
    with col_f2:
        end_date = st.date_input("End date", value=date.today())
    st.caption("Filters apply to the dashboard and transaction table.")
    st.divider()

    st.subheader("Quick Export")
    if st.button("Export current view to CSV", use_container_width=True):
        df_export = fetch_transactions(user_filter=view_user, start=start_date, end=end_date)
        if df_export.empty:
            st.info("No data to export for the current filters.")
        else:
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv, file_name="family_finance_export.csv", mime="text/csv", use_container_width=True)

# Main: Entry Forms & Dashboard
tabs = st.tabs(["➕ Add Transaction", "📊 Dashboard", "📜 Transactions"])

# ---- Add Transaction Tab ----
with tabs[0]:
    st.subheader("Add a new transaction")
    # Form split
    c1, c2 = st.columns([2, 3])
    with c1:
        ttype = st.radio("Type", ["Expense", "Income"], horizontal=True)
        # dynamic categories
        cats = DEFAULT_CATEGORIES["Expense" if ttype == "Expense" else "Income"]
        category = st.selectbox("Category", options=cats)
        amount = st.number_input("Amount", min_value=0.0, step=1.0, format="%.2f")
    with c2:
        tdate = st.date_input("Date", value=date.today())
        description = st.text_input("Description (optional)", placeholder="e.g., Grocery run at Walmart")

    col_b1, col_b2, _ = st.columns([1, 1, 2])
    with col_b1:
        if st.button("Save Transaction", type="primary", use_container_width=True, disabled=amount <= 0):
            insert_transaction(user, ttype, category, amount, tdate, description)
            st.success("Transaction saved.")
    with col_b2:
        if st.button("Add Another (prefill date)", use_container_width=True):
            if amount > 0:
                insert_transaction(user, ttype, category, amount, tdate, description)
                st.success("Transaction added. You can continue adding more.")
            else:
                st.info("Enter a valid amount first.")

# ---- Dashboard Tab ----
with tabs[1]:
    st.subheader("Overview")
    df_view = fetch_transactions(user_filter=view_user, start=start_date, end=end_date)

    k1, k2, k3 = st.columns(3)
    inc, exp, net = summary_metrics(df_view)
    with k1: kpi_card("Total Income", inc)
    with k2: kpi_card("Total Expenses", exp)
    with k3: kpi_card("Net", net)
    divider()

    # Charts
    st.markdown("#### Trends by Month")
    roll = monthly_rollup(df_view)
    if roll.empty:
        st.info("No data for the selected period.")
    else:
        fig1 = px.bar(
            roll, x="month", y="amount", color="ttype",
            barmode="group",
            title="Monthly Totals by Type"
        )
        fig1.update_layout(margin=dict(l=10,r=10,t=40,b=10), xaxis_title=None, yaxis_title="Amount ($)")
        st.plotly_chart(fig1, use_container_width=True, theme="streamlit")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### This Month by Category (Expenses)")
        if not df_view.empty:
            # FIX: no `.dt` on scalar Period
            this_month = pd.to_datetime(date.today()).to_period("M").to_timestamp()
            cat_roll = category_rollup(df_view, ttype_filter="Expense", month=this_month)
            if cat_roll.empty:
                st.info("No expenses this month.")
            else:
                fig2 = px.pie(cat_roll, names="category", values="amount", title="Current Month Expense Breakdown")
                fig2.update_layout(margin=dict(l=10,r=10,t=40,b=10))
                st.plotly_chart(fig2, use_container_width=True, theme="streamlit")
        else:
            st.info("No data available.")

    with col_c2:
        st.markdown("#### Top Categories (All Time / Filtered Range)")
        top_cat = category_rollup(df_view, ttype_filter=None)
        if top_cat.empty:
            st.info("No data to display.")
        else:
            fig3 = px.bar(top_cat.head(10), x="amount", y="category", orientation="h", title="Top 10 Categories")
            fig3.update_layout(margin=dict(l=10,r=10,t=40,b=10), xaxis_title="Amount ($)", yaxis_title=None)
            st.plotly_chart(fig3, use_container_width=True, theme="streamlit")

# ---- Transactions Tab ----
with tabs[2]:
    st.subheader("All Transactions (filtered)")
    df = fetch_transactions(user_filter=view_user, start=start_date, end=end_date)
    if df.empty:
        st.info("No transactions found for the selected filters.")
    else:
        # Pretty columns
        df_display = df.copy()
        df_display.rename(columns={
            "id":"ID", "user":"User", "ttype":"Type", "category":"Category",
            "amount":"Amount ($)", "tdate":"Date", "description":"Description"
        }, inplace=True)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        st.markdown("##### Delete Selected")
        ids_to_delete = st.multiselect("Choose transaction IDs to delete", df["id"].tolist())
        del_col1, del_col2 = st.columns([1, 5])
        with del_col1:
            if st.button("Delete", type="secondary", disabled=len(ids_to_delete) == 0):
                delete_transactions(ids_to_delete)
                st.success(f"Deleted {len(ids_to_delete)} transaction(s). Refresh the tab to see changes.")

# ============ Footer ============
st.divider()
st.caption(
    "Built with ❤️ in Streamlit. Data is saved locally in 'family_finance.db'. "
    "You can back it up by copying the file alongside this app."
)
