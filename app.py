
import os
import glob
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

st.set_page_config(page_title="Kyrk – Race Results Q&A", layout="wide")

# ── API key ─────────────────────────────────────────────────────────────
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_key:
    st.error(
        "Add your OpenAI key:\n"
        " • set the env-var OPENAI_API_KEY\n"
        "   OR\n"
        " • add it in Streamlit ▸ Settings ▸ Secrets"
    )
    st.stop()

# ── Locate CSV ─────────────────────────────────────────────────────────
DEFAULT_CSV = "KYRK_RESULTS.csv"
csv_file = None
if os.path.exists(DEFAULT_CSV):
    csv_file = DEFAULT_CSV
else:
    # first CSV in repo root
    for f in glob.glob("*.csv"):
        csv_file = f
        break

if csv_file is None:
    st.error(
        "No CSV found next to app.py. Rename your file to 'KYRK_RESULTS.csv' "
        "or upload it via the widget below."
    )
    uploaded = st.file_uploader("Upload race results CSV", type="csv")
    if not uploaded:
        st.stop()
    csv_file = uploaded

# ── Load & tidy data ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file):
    df = pd.read_csv(file)
    # Try to build a case‑folded name column if we can spot a name field
    for name_col in ["NAME", "Name", "Runner", "Athlete"]:
        if name_col in df.columns:
            df["NAME_CLEAN"] = df[name_col].astype(str).str.strip().str.casefold()
            break

    # Convert any obvious numeric columns
    for col in df.columns:
        if re.search(r"laps?|distance|time|hours|mins?|seconds?", col, re.I):
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

DF = load_data(csv_file)

# ── LLM wrapper ─────────────────────────────────────────────────────────
llm = OpenAI(model_name="gpt-4o-mini-2025-04", api_key=openai_key, temperature=0)
sdf = SmartDataframe(DF, config={"llm": llm, "verbose": False})

# ── UI ──────────────────────────────────────────────────────────────────
st.title("🏃 Kyrk – Ask Anything about the Race Results")

query = st.text_input(
    "Ask a question about the results:",
    placeholder="Who won 2024 and how many laps?",
    key="nl_query"
)

if st.button("Submit") and query:
    with st.spinner("Thinking…"):
        try:
            result = sdf.chat(query)

            import pandas as pd
            # unwrap helper types
            if hasattr(result, "to_pandas"):
                result = result.to_pandas()

            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
                csv_bytes = result.to_csv(index=False).encode()
                st.download_button(
                    "Download table as CSV",
                    csv_bytes,
                    "answer.csv",
                    "text/csv"
                )
            elif isinstance(result, (list, tuple, set)):
                st.write(", ".join(map(str, result)))
            else:
                st.write(result)

        except Exception as e:
            st.error(str(e))
