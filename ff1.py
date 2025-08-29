# --- Hugging Face Spaces hardening: cache/config to /tmp to avoid permission issues ---
import os
import pathlib
import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, date
from contextlib import contextmanager
import threading
from pathlib import Path

# Safe plotly import with fallback
PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    st.error("⚠️ Plotly is required for charts. Please add 'plotly>=5.17.0' to your requirements.txt file.")
    st.info("The app will continue to work, but charts will not be available.")
    # Create dummy px object to prevent errors
    class DummyPlotly:
        def bar(self, *args, **kwargs):
            return None
        def pie(self, *args, **kwargs): 
            return None
    px = DummyPlotly()

# Environment setup for cloud deployment
for var, sub in {
    "MPLCONFIGDIR": "matplotlib",
    "XDG_CACHE_HOME": "xdg", 
    "HF_HOME": "hf",
    "PIP_CACHE_DIR": "pip",
    "HUGGINGFACE_HUB_CACHE": "hfhub",
    "NUMBA_CACHE_DIR": "numba",
}.items():
    try:
        d = pathlib.Path("/tmp") / sub
        d.mkdir(parents=True, exist_ok=True)
        os.environ[var] = str(d)
    except Exception:
        pass

# Streamlit specific stability tweaks
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "poll")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")

# ============ App Config ============
st.set_page_config(
    page_title="Family Finance Manager",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============ Constants ============
# Use /tmp for cloud deployment to avoid permission issues
DB_PATH = "/tmp/family_finance.db" if os.path.exists("/tmp") else "family_finance.db"
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

# Thread lock for database operations
db_lock = threading.Lock()

# ============ DB Helpers ============
@contextmanager
def get_conn():
    """Thread-safe database connection with proper error handling"""
    with db_lock:
        conn = None
        try:
            conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30.0)
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")  # Better for concurrent access
            yield conn
        except sqlite3.Error as e:
            st.error(f"Database error: {e}")
            if conn:
                conn.rollback()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                try:
                    conn.commit()
                    conn.close()
                except:
                    pass

def init_db():
    """Initialize database with proper error handling"""
    try:
        with get_conn() as conn:
            if conn:
                conn.execute(
                    '''
                    CREATE TABLE IF NOT EXISTS transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user TEXT NOT NULL,
                        ttype TEXT CHECK(ttype IN ('Income','Expense')) NOT NULL,
                        category TEXT NOT NULL,
                        amount REAL NOT NULL CHECK(amount > 0),
                        tdate TEXT NOT NULL,
                        description TEXT DEFAULT '',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    '''
                )
                # Create index for better performance
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_user_date ON transactions(user, tdate);"
                )
    except Exception as e:
        st.error(f"Failed to initialize database: {e}")

def insert_transaction(user, ttype, category, amount, tdate, description):
    """Insert transaction with validation and error handling"""
    try:
        # Input validation
        if not user or user not in APP_USERS:
            st.error("Invalid user")
            return False
        if not ttype or ttype not in ["Income", "Expense"]:
            st.error("Invalid transaction type")
            return False
        if not category:
            st.error("Category is required")
            return False
        if amount <= 0:
            st.error("Amount must be greater than 0")
            return False
        
        with get_conn() as conn:
            if conn:
                conn.execute(
                    """INSERT INTO transactions 
                       (user, ttype, category, amount, tdate, description) 
                       VALUES (?, ?, ?, ?, ?, ?);""",
                    (user, ttype, category, float(amount), tdate.isoformat(), description or "")
                )
                return True
    except Exception as e:
        st.error(f"Failed to save transaction: {e}")
    return False

def delete_transactions(ids):
    """Delete transactions with validation"""
    if not ids:
        return False
    try:
        with get_conn() as conn:
            if conn:
                qmarks = ",".join("?" for _ in ids)
                cursor = conn.execute(f"DELETE FROM transactions WHERE id IN ({qmarks});", ids)
                deleted_count = cursor.rowcount
                return deleted_count > 0
    except Exception as e:
        st.error(f"Failed to delete transactions: {e}")
    return False

def fetch_transactions(user_filter=None, start=None, end=None):
    """Fetch transactions with improved error handling"""
    try:
        with get_conn() as conn:
            if not conn:
                return pd.DataFrame()
                
            query = """SELECT id, user, ttype, category, amount, tdate, description 
                      FROM transactions WHERE 1=1"""
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
                
            query += " ORDER BY date(tdate) DESC, id DESC LIMIT 1000;"  # Limit for performance
            
            df = pd.read_sql_query(query, conn, params=params)
            
        if not df.empty:
            # Safe datetime conversion
            df["tdate"] = pd.to_datetime(df["tdate"], errors="coerce")
            df = df.dropna(subset=["tdate"])
            df["tdate"] = df["tdate"].dt.date
            
        return df
    except Exception as e:
        st.error(f"Failed to fetch transactions: {e}")
        return pd.DataFrame()

def summary_metrics(df):
    """Calculate summary metrics safely"""
    try:
        if df.empty:
            return 0.0, 0.0, 0.0
        
        total_income = df.loc[df["ttype"] == "Income", "amount"].sum()
        total_expense = df.loc[df["ttype"] == "Expense", "amount"].sum()
        net = total_income - total_expense
        
        return float(total_income), float(total_expense), float(net)
    except Exception as e:
        st.error(f"Error calculating metrics: {e}")
        return 0.0, 0.0, 0.0

def monthly_rollup(df):
    """Create monthly rollup with error handling"""
    try:
        if df.empty or "tdate" not in df.columns:
            return pd.DataFrame(columns=["month", "ttype", "amount"])
            
        temp = df.copy()
        temp["tdate"] = pd.to_datetime(temp["tdate"], errors="coerce")
        temp = temp.dropna(subset=["tdate"])
        
        if temp.empty:
            return pd.DataFrame(columns=["month", "ttype", "amount"])
            
        temp["month"] = temp["tdate"].dt.to_period("M").dt.to_timestamp()
        grp = temp.groupby(["month", "ttype"], as_index=False)["amount"].sum()
        return grp
    except Exception as e:
        st.error(f"Error creating monthly rollup: {e}")
        return pd.DataFrame(columns=["month", "ttype", "amount"])

def category_rollup(df, ttype_filter="Expense", month=None):
    """Create category rollup with error handling"""
    try:
        if df.empty:
            return pd.DataFrame(columns=["category", "amount"])
            
        temp = df.copy()
        temp["tdate"] = pd.to_datetime(temp["tdate"], errors="coerce")
        temp = temp.dropna(subset=["tdate"])
        
        if temp.empty:
            return pd.DataFrame(columns=["category", "amount"])
            
        if month is not None:
            temp["month"] = temp["tdate"].dt.to_period("M").dt.to_timestamp()
            temp = temp[temp["month"] == month]
            
        if ttype_filter:
            temp = temp[temp["ttype"] == ttype_filter]
            
        grp = temp.groupby(["category"], as_index=False)["amount"].sum()
        grp = grp.sort_values("amount", ascending=False)
        return grp
    except Exception as e:
        st.error(f"Error creating category rollup: {e}")
        return pd.DataFrame(columns=["category", "amount"])

# ============ UI Helpers ============
def kpi_card(label, value):
    """Display KPI card with proper formatting"""
    color = "normal"
    if "Net" in label:
        color = "normal" if value >= 0 else "inverse"
    st.metric(label, f"${value:,.2f}")

def safe_plotly_chart(fig, **kwargs):
    """Safely display plotly chart with error handling"""
    if not PLOTLY_AVAILABLE or fig is None:
        st.warning("📊 Charts are not available. Please install plotly: `pip install plotly>=5.17.0`")
        return
    try:
        st.plotly_chart(fig, **kwargs)
    except Exception as e:
        st.error(f"Error displaying chart: {e}")

def create_fallback_chart(data, chart_type="table"):
    """Create fallback visualization when plotly is not available"""
    if chart_type == "table":
        st.dataframe(data, use_container_width=True)
    elif chart_type == "metrics" and not data.empty:
        for _, row in data.head(5).iterrows():
            st.metric(str(row.iloc[0]), f"${row.iloc[1]:,.2f}" if len(row) > 1 else str(row.iloc[1]))

# ============ App Initialization ============
# Initialize database
try:
    init_db()
except Exception as e:
    st.error(f"Failed to initialize app: {e}")
    st.stop()

# Add session state for better UX
if "refresh_data" not in st.session_state:
    st.session_state.refresh_data = 0

# ============ Sidebar ============
with st.sidebar:
    st.title("💸 Family Finance Manager")
    st.caption("Track income & expenses for George and Jane.")
    
    # User selection with validation
    try:
        user = st.selectbox(
            "Active user", 
            APP_USERS, 
            index=0, 
            help="Entries you add will be recorded under this name."
        )
    except:
        user = APP_USERS[0]
        
    st.divider()

    st.subheader("Filters")
    view_user = st.selectbox("View data for", ["All"] + APP_USERS, index=0)
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        try:
            start_date = st.date_input(
                "Start date", 
                value=date(date.today().year, 1, 1),
                max_value=date.today()
            )
        except:
            start_date = date(date.today().year, 1, 1)
            
    with col_f2:
        try:
            end_date = st.date_input(
                "End date", 
                value=date.today(),
                max_value=date.today()
            )
        except:
            end_date = date.today()
    
    if start_date > end_date:
        st.error("Start date must be before end date")
        start_date = date(date.today().year, 1, 1)
        
    st.caption("Filters apply to the dashboard and transaction table.")
    st.divider()

    st.subheader("Quick Export")
    if st.button("Export current view to CSV", use_container_width=True):
        df_export = fetch_transactions(user_filter=view_user, start=start_date, end=end_date)
        if df_export.empty:
            st.info("No data to export for the current filters.")
        else:
            try:
                csv = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV", 
                    csv, 
                    file_name=f"family_finance_export_{datetime.now().strftime('%Y%m%d')}.csv", 
                    mime="text/csv", 
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to generate CSV: {e}")

# ============ Main App ============
tabs = st.tabs(["➕ Add Transaction", "📊 Dashboard", "📜 Transactions"])

# ---- Add Transaction Tab ----
with tabs[0]:
    st.subheader(f"Add a new transaction for {user}")
    
    with st.form("transaction_form", clear_on_submit=True):
        c1, c2 = st.columns([2, 3])
        
        with c1:
            ttype = st.radio("Type", ["Expense", "Income"], horizontal=True)
            cats = DEFAULT_CATEGORIES.get(ttype, [])
            category = st.selectbox("Category", options=cats, index=0 if cats else None)
            amount = st.number_input(
                "Amount", 
                min_value=0.01, 
                max_value=1000000.0,
                step=1.0, 
                format="%.2f"
            )
            
        with c2:
            tdate = st.date_input(
                "Date", 
                value=date.today(),
                max_value=date.today()
            )
            description = st.text_input(
                "Description (optional)", 
                placeholder="e.g., Grocery run at Walmart",
                max_chars=200
            )

        submitted = st.form_submit_button("Save Transaction", type="primary")
        
        if submitted:
            if amount > 0 and category:
                success = insert_transaction(user, ttype, category, amount, tdate, description)
                if success:
                    st.success("✅ Transaction saved successfully!")
                    st.session_state.refresh_data += 1
            else:
                st.error("Please fill in all required fields with valid values.")

# ---- Dashboard Tab ----  
with tabs[1]:
    st.subheader("Financial Overview")
    
    # Auto-refresh data
    df_view = fetch_transactions(user_filter=view_user, start=start_date, end=end_date)

    # KPI Cards
    k1, k2, k3 = st.columns(3)
    inc, exp, net = summary_metrics(df_view)
    
    with k1: 
        kpi_card("Total Income", inc)
    with k2: 
        kpi_card("Total Expenses", exp)
    with k3: 
        kpi_card("Net Balance", net)

    st.divider()

    # Charts
    st.markdown("#### Monthly Trends")
    roll = monthly_rollup(df_view)
    
    if roll.empty:
        st.info("📈 No data available for the selected period. Add some transactions to see trends!")
    else:
        if PLOTLY_AVAILABLE:
            try:
                fig1 = px.bar(
                    roll, 
                    x="month", 
                    y="amount", 
                    color="ttype",
                    barmode="group",
                    title="Monthly Income vs Expenses",
                    color_discrete_map={"Income": "#2E8B57", "Expense": "#DC143C"}
                )
                fig1.update_layout(
                    margin=dict(l=10, r=10, t=40, b=10), 
                    xaxis_title="Month", 
                    yaxis_title="Amount ($)"
                )
                safe_plotly_chart(fig1, use_container_width=True, theme="streamlit")
            except Exception as e:
                st.error(f"Could not display monthly trends: {e}")
                create_fallback_chart(roll, "table")
        else:
            st.subheader("Monthly Trends (Table View)")
            create_fallback_chart(roll, "table")

    # Category Breakdowns
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.markdown("#### Current Month Expenses")
        if not df_view.empty:
            try:
                this_month = pd.to_datetime(date.today()).to_period("M").to_timestamp()
                cat_roll = category_rollup(df_view, ttype_filter="Expense", month=this_month)
                
                if cat_roll.empty:
                    st.info("No expenses recorded this month.")
                else:
                    if PLOTLY_AVAILABLE:
                        fig2 = px.pie(
                            cat_roll, 
                            names="category", 
                            values="amount", 
                            title="Expense Categories This Month"
                        )
                        fig2.update_layout(margin=dict(l=10, r=10, t=40, b=10))
                        safe_plotly_chart(fig2, use_container_width=True, theme="streamlit")
                    else:
                        create_fallback_chart(cat_roll, "table")
            except Exception as e:
                st.error(f"Could not display category breakdown: {e}")
        else:
            st.info("No data available for current month.")

    with col_c2:
        st.markdown("#### Top Spending Categories")
        top_cat = category_rollup(df_view, ttype_filter="Expense")
        
        if top_cat.empty:
            st.info("No expense data to display.")
        else:
            if PLOTLY_AVAILABLE:
                try:
                    fig3 = px.bar(
                        top_cat.head(10), 
                        x="amount", 
                        y="category", 
                        orientation="h",
                        title="Top 10 Expense Categories"
                    )
                    fig3.update_layout(
                        margin=dict(l=10, r=10, t=40, b=10), 
                        xaxis_title="Amount ($)", 
                        yaxis_title=None
                    )
                    safe_plotly_chart(fig3, use_container_width=True, theme="streamlit")
                except Exception as e:
                    st.error(f"Could not display top categories: {e}")
                    create_fallback_chart(top_cat.head(10), "table")
            else:
                create_fallback_chart(top_cat.head(10), "table")

# ---- Transactions Tab ----
with tabs[2]:
    st.subheader("Transaction History")
    
    df = fetch_transactions(user_filter=view_user, start=start_date, end=end_date)
    
    if df.empty:
        st.info("📝 No transactions found for the selected filters. Try adjusting your date range or user filter.")
    else:
        # Display transaction count
        st.caption(f"Showing {len(df)} transaction(s)")
        
        # Pretty display
        df_display = df.copy()
        df_display.rename(columns={
            "id": "ID", 
            "user": "User", 
            "ttype": "Type", 
            "category": "Category",
            "amount": "Amount ($)", 
            "tdate": "Date", 
            "description": "Description"
        }, inplace=True)
        
        # Format amount column
        df_display["Amount ($)"] = df_display["Amount ($)"].apply(lambda x: f"${x:,.2f}")
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        # Delete functionality (admin-like feature)
        with st.expander("🗑️ Delete Transactions (Advanced)"):
            st.warning("⚠️ This action cannot be undone!")
            ids_to_delete = st.multiselect(
                "Select transaction IDs to delete", 
                df["id"].tolist(),
                help="Select one or more transaction IDs from the table above"
            )
            
            if ids_to_delete:
                st.write(f"You are about to delete {len(ids_to_delete)} transaction(s).")
                
            col_del1, col_del2 = st.columns([1, 4])
            with col_del1:
                if st.button(
                    f"Delete {len(ids_to_delete)} Transaction(s)", 
                    type="secondary", 
                    disabled=len(ids_to_delete) == 0
                ):
                    success = delete_transactions(ids_to_delete)
                    if success:
                        st.success(f"✅ Successfully deleted {len(ids_to_delete)} transaction(s)!")
                        st.session_state.refresh_data += 1
                        st.rerun()
                    else:
                        st.error("Failed to delete transactions.")

# ============ Footer ============
st.divider()
st.caption(
    "🔒 **Privacy Note**: This is a shared demo. In production, implement proper user authentication. "
    "Data is stored temporarily and may be reset periodically."
)

# Add refresh button for better UX
if st.button("🔄 Refresh Data"):
    st.session_state.refresh_data += 1
    st.rerun()
