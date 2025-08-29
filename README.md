---
title: Family Finance Manager
emoji: 💸
colorFrom: indigo
colorTo: emerald
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# Family Finance Manager

A simple Streamlit app that tracks income & expenses with a local SQLite database and shows interactive Plotly dashboards.

## How to run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Notes

- On Hugging Face Spaces, the SQLite file (`family_finance.db`) is created in the Space's working directory.  
- If you want data to persist across Space restarts, enable **Persistent Storage** in your Space settings.
