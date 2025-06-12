import os, glob, re
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

st.set_page_config(page_title="Kyrk – Race Results Q&A", layout="wide")

# ── API key ─────────────────────────────────────────────────────────────
openai_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not openai_key:
    st.error("Add OPENAI_API_KEY via environment variable or Streamlit Secrets.")
    st.stop()

# ── Locate CSV ─────────────────────────────────────────────────────────
CSV_NAME = "KYRK_RESULTS.csv"
if os.path.exists(CSV_NAME):
    csv_path = CSV_NAME
else:
    csv_candidates = glob.glob("*.csv")
    csv_path = csv_candidates[0] if csv_candidates else None

if csv_path is None:
    upload = st.file_uploader("Upload race results CSV", type="csv")
    if not upload:
        st.stop()
    csv_path = upload

# ── Load & tidy data ────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file_like):
    df = pd.read_csv(file_like)
    # Best‑guess canonical name column
    for col in ["NAME", "Name", "Runner", "Athlete"]:
        if col in df.columns:
            df["NAME_CLEAN"] = df[col].astype(str).str.strip().str.casefold()
            break
    # numeric coercions
    for col in df.columns:
        if re.search(r"laps|distance|time|hours|mins|seconds", col, re.I):
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df

DF = load_data(csv_path)

# ── LLM wrapper ─────────────────────────────────────────────────────────
llm = OpenAI(model_name="gpt-4o-mini-2025-04", api_key=openai_key, temperature=0)
sdf = SmartDataframe(
    DF,
    config={
        "llm": llm,
        "verbose": False,
        "system_prompt": "\nYou are a data assistant for endurance-race results stored in a pandas DataFrame.\n\nDATA ASSUMPTIONS\n----------------\n* Each row is one runner\u2019s performance in one edition of the race.\n* Key columns (case\u2011sensitive):\n    - YEAR   (or YEAR OF RACE)    \u2192 integer\n    - Gender                       \u2192 'm' or 'f'\n    - NAME (or Runner, Athlete)    \u2192 full name\n    - POSITION                     \u2192 integer rank or the string 'Winner'\n    - TIME or FINISH_TIME          \u2192 string HH:MM:SS\n    - LAPS_COMPLETED or LAPS       \u2192 numeric\n\nRESPONSE GUIDELINES\n-------------------\n1. **Winners / podium**  \n   \u2022 If a question mentions winners, podium, fastest, etc. **and the user does NOT specify gender**, always return results for *both male and female* categories.  \n   \u2022 Display them side\u2011by\u2011side in the same DataFrame (columns: YEAR, Gender, NAME, POSITION, TIME/LAPS).\n\n2. **Counts by sex / category**  \n   \u2022 Questions like \u201cHow many women finished each year?\u201d \u2192 group by YEAR and return rows for every year, not a single aggregate.  \n   \u2022 Column names: YEAR, COUNT.\n\n3. **Top\u2011N lists with a gender adjective** (e.g. \u201ctop 3 female finishers\u201d) \u2192 filter `Gender == 'f'`.\n\n4. Prefer DataFrame outputs where practical so the UI can render a nice table.\n\n5. Use the exact column names present in the loaded DataFrame.\n"
    }
)

# ── UI ──────────────────────────────────────────────────────────────────
st.title("🏃 Kyrk – Ask Anything about the Race Results")

query = st.text_input(
    "Ask a question about the results:",
    placeholder="Who were the male and female winners in 2022?"
)

if st.button("Submit") and query:
    with st.spinner("Crunching…"):
        try:
            result = sdf.chat(query)
            # unwrap custom pandas‑ai objects
            if hasattr(result, "to_pandas"):
                result = result.to_pandas()
            # Pretty display
            if isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download as CSV",
                    result.to_csv(index=False).encode(),
                    "answer.csv",
                    "text/csv"
                )
            elif isinstance(result, (list, tuple, set)):
                st.write(", ".join(map(str, result)))
            else:
                st.write(result)
        except Exception as e:
            st.error(str(e))