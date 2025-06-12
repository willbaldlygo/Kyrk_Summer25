
# Kyrk – Race Results Q&A

Streamlit app that lets you query the Kyrk race results CSV in natural language,
powered by **pandas-ai** + GPT.

## Quick start (local)

```bash
git clone https://github.com/your-user/kyrk-app.git
cd kyrk-app
pip install -r requirements.txt

# Put your CSV in the repo root
mv "ALL SUMMER SPINE RACE RESULTS - ALL SUMMER SPINE RESULTS.csv" KYRK_RESULTS.csv

export OPENAI_API_KEY=sk-...

streamlit run app.py
```

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. In Streamlit Cloud create a new app pointing to `app.py`.
3. Add your `OPENAI_API_KEY` in **Settings ▸ Secrets**:

   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

4. Commit the CSV as `KYRK_RESULTS.csv` (or upload through the app).

The deployed URL will let anyone ask questions like **"Which runners have gone further than 35 laps?"** and receive a nicely formatted table.

## Notes

* Runtime pinned to **Python 3.12** to ensure wheels exist for pandas/NumPy.
* `PyYAML` is included because pandasai requires it at runtime.
* If you don't want to include raw results in the repo, delete `KYRK_RESULTS.csv` and let users upload via the widget.
