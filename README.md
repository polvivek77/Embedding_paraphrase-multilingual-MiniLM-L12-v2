# Text → Numbers: Embeddings (Streamlit Demo)

A beginner-friendly app that shows real tokenizer pieces, token IDs, and a real sentence embedding from `paraphrase-multilingual-MiniLM-L12-v2`.

## Requirements
- Python 3.10–3.12 recommended
- Internet connection for the first model download

## Run on Windows

1. Extract the ZIP file.
2. Open the `embedding_streamlit_demo` folder.
3. Open Command Prompt or PowerShell in this folder.
4. Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

5. Install packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

6. Start the app:

```bash
streamlit run app.py
```

7. Streamlit should open a browser tab, usually at `http://localhost:8501`.

## What to try
- Enter `I love mangoes`, then click **Convert text to numbers**.
- Inspect the token table and embedding vector.
- Expand **Show the complete embedding vector**.
- Compare `I love mangoes` with `I really enjoy eating mangoes`, then compare it with `The train arrives at noon`.

## Important beginner notes
- Token IDs are labels/identifiers; they do not represent the meaning of a word.
- The final sentence embedding is a list of 384 real numbers for this model.
- The bar chart displays the first 40 dimensions only.
- The model download happens on first use and requires internet access.
