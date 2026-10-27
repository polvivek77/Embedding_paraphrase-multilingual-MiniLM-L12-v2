import logging
import time
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MODEL_URL = f"https://huggingface.co/sentence-transformers/{MODEL_NAME}"
MODEL_LICENSE = "Apache-2.0"
COST_HELP = (
    "This open-source model runs locally on this machine, so there are no API or per-token charges. "
    "The only cost is your own CPU/GPU time."
)
LOG_FILE = Path(__file__).resolve().parent / "app.log"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | session=%(session_id)s | %(name)s:%(lineno)d | %(message)s"


class _SessionIdDefault(logging.Filter):
    # Library log records have no session id; give them a placeholder so the format string works.
    def filter(self, record):
        if not hasattr(record, "session_id"):
            record.session_id = "-"
        return True


def setup_logging():
    # Streamlit re-executes this script on every interaction, so only attach handlers once per process.
    logger = logging.getLogger("embedding_app")
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(_SessionIdDefault())

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(_SessionIdDefault())

    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    # Record sentence-transformers' own model-loading messages in app.log too.
    st_logger = logging.getLogger("sentence_transformers")
    st_logger.setLevel(logging.INFO)
    st_logger.addHandler(file_handler)

    logger.info("Logging initialised; writing to %s", LOG_FILE)
    return logger


logger = setup_logging()

st.set_page_config(page_title="Text → Numbers: Embeddings", page_icon="🧠", layout="wide")

is_new_session = "session_id" not in st.session_state
if is_new_session:
    st.session_state["session_id"] = uuid.uuid4().hex[:8]
log = logging.LoggerAdapter(logger, {"session_id": st.session_state["session_id"]})
if is_new_session:
    log.info("New browser session started")
log.debug("Script run started | converted=%s", st.session_state.get("converted", False))


@st.cache_resource
def load_model():
    # Downloads the model the first time you run the app. Returns the model plus its specs for the model card.
    logger.info("Loading embedding model '%s' (downloads on first use)", MODEL_NAME)
    start = time.perf_counter()
    model = SentenceTransformer(MODEL_NAME)
    load_seconds = time.perf_counter() - start

    transformer = model[0].auto_model
    parameters = list(model.parameters())
    specs = {
        "dimensions": model.get_embedding_dimension(),
        "max_seq_length": model.max_seq_length,
        "parameters": sum(p.numel() for p in parameters),
        "size_mb": sum(p.numel() * p.element_size() for p in parameters) / 1024**2,
        "dtype": str(parameters[0].dtype).replace("torch.", ""),
        "architecture": type(transformer).__name__,
        "layers": transformer.config.num_hidden_layers,
        "attention_heads": transformer.config.num_attention_heads,
        "hidden_size": transformer.config.hidden_size,
        "vocab_size": model.tokenizer.vocab_size,
        "pooling": model[1].pooling_mode,
        "normalized": any(type(module).__name__ == "Normalize" for module in model),
        "similarity": model.similarity_fn_name,
        "device": str(model.device),
        "load_seconds": load_seconds,
    }
    logger.info("Model '%s' loaded in %.2fs | specs=%s", MODEL_NAME, load_seconds, specs)
    return model, specs


st.title("🧠 Text → Numbers: See Embeddings in Action")
st.markdown(f":material/memory: **Embedding model in use:** [`{MODEL_NAME}`]({MODEL_URL}) (sentence-transformers)")
st.write(
    "Type text and explore how an embedding model turns it into a list of numbers. "
    "The first run may take a little while because the model needs to download."
)

with st.expander("🔎 The journey in one glance", expanded=True):
    st.markdown(
        "**Your sentence** → **tokens** → **token IDs** → "
        "**learned vector for each token (model internals)** → **sentence embedding**"
    )
    st.caption(
        "Note: this app shows real token IDs from the model's tokenizer and the real final embedding. "
        "The model's internal token vectors are not the same as token IDs."
    )

default_text = "I love mangoes"
text = st.text_input("✍️ Enter a sentence", value=default_text, max_chars=500)

with st.sidebar:
    st.header(":material/memory: Model in use")
    st.markdown(f"[`{MODEL_NAME}`]({MODEL_URL})")
    try:
        with st.spinner("Loading the model..."):
            _, specs = load_model()
    except Exception:
        log.exception("Failed to load model '%s' for the model card", MODEL_NAME)
        st.error(f"Could not load the model. Details were written to `{LOG_FILE.name}`.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Dimensions", specs["dimensions"], border=True, help="Length of the vector produced for each sentence.")
        m2.metric("Max input", f"{specs['max_seq_length']} tokens", border=True, help="Longer inputs are truncated.")
        m3, m4 = st.columns(2)
        m3.metric("Parameters", f"{specs['parameters'] / 1e6:.1f}M", border=True)
        m4.metric("Weights size", f"{specs['size_mb']:.0f} MB", border=True, help="Size of the model weights in memory.")
        m5, m6 = st.columns(2)
        m5.metric("Cost", "$0.00", border=True, help=COST_HELP)
        m6.metric("Load time", f"{specs['load_seconds']:.1f} s", border=True, help="Time this server took to load the model into memory.")

        details = pd.DataFrame(
            [
                ("Architecture", f"{specs['architecture']} · {specs['layers']} layers · {specs['attention_heads']} heads"),
                ("Hidden size", str(specs["hidden_size"])),
                ("Vocabulary", f"{specs['vocab_size']:,} tokens"),
                ("Pooling", specs["pooling"]),
                ("Normalized output", "Yes" if specs["normalized"] else "No"),
                ("Similarity function", specs["similarity"]),
                ("Weights dtype", specs["dtype"]),
                ("Running on", specs["device"]),
                ("License", MODEL_LICENSE),
                ("Language", "English"),
            ],
            columns=["Property", "Value"],
        )
        st.dataframe(details, hide_index=True, alt=f"Technical details of the {MODEL_NAME} model")

if st.button("✨ Convert text to numbers", type="primary"):
    log.info("Convert clicked | text_length=%d", len(text))
    log.debug("Input text: %r", text)
    if not text.strip():
        log.warning("Convert aborted: input text is empty")
        st.warning("Please enter some text first.")
        st.stop()

    try:
        with st.spinner("Loading the model and creating your embedding..."):
            start = time.perf_counter()
            model, specs = load_model()
            tokenizer = model.tokenizer

            # Token IDs are real IDs assigned by this model's tokenizer.
            # max_length matches the model, so the table shows exactly the tokens the model sees.
            encoded = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=model.max_seq_length,
                return_tensors="pt"
            )
            tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].tolist())
            token_ids = encoded["input_ids"][0].tolist()
            vector = np.asarray(model.encode(text, normalize_embeddings=False), dtype=float)
            elapsed = time.perf_counter() - start
    except Exception:
        log.exception("Failed to create embedding for input text")
        st.error(f"Something went wrong while creating the embedding. Details were written to `{LOG_FILE.name}`.")
        st.stop()

    log.info(
        "Embedding created in %.3fs | tokens=%d | dimensions=%d | l2_norm=%.4f",
        elapsed, len(tokens), len(vector), float(np.linalg.norm(vector)),
    )
    log.debug("Tokens: %s", tokens)
    log.debug("Token IDs: %s", token_ids)
    log.debug("Embedding (first 10 values): %s", np.round(vector[:10], 6).tolist())

    st.session_state["tokens"] = tokens
    st.session_state["token_ids"] = token_ids
    st.session_state["vector"] = vector
    st.session_state["embed_seconds"] = elapsed
    st.session_state["max_tokens"] = specs["max_seq_length"]
    st.session_state["converted_text"] = text
    st.session_state["converted"] = True

if st.session_state.get("converted", False):
    tokens = st.session_state["tokens"]
    token_ids = st.session_state["token_ids"]
    vector = st.session_state["vector"]
    st.success(f"Done! Embedding created with `{MODEL_NAME}` for: “{st.session_state['converted_text']}”")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Embedding dimensions", len(vector), border=True)
    s2.metric(
        "Tokens used", f"{len(tokens)} / {st.session_state['max_tokens']}", border=True,
        help="Tokens in your sentence (including the special [CLS] and [SEP] markers) out of the model's maximum.",
    )
    s3.metric(
        "Time taken", f"{st.session_state['embed_seconds'] * 1000:.0f} ms", border=True,
        help="Tokenization plus embedding time for this sentence. The first run also includes loading the model.",
    )
    s4.metric("Cost", "$0.00", border=True, help=COST_HELP)

    st.subheader("1️⃣ Tokenization + Token IDs")
    st.write("The tokenizer splits the text into pieces. Each piece gets a numeric ID.")
    token_df = pd.DataFrame({
        "Token (text piece)": tokens,
        "Token ID": token_ids
    })
    st.dataframe(token_df, width="stretch", hide_index=True)
    st.caption("Token pieces may be whole words, parts of words, punctuation, or special markers.")

    st.subheader("2️⃣ The final sentence embedding")
    st.write(
        f"The model `{MODEL_NAME}` represents your entire sentence using **{len(vector)} numbers**. "
        "These numbers are learned by the model during training."
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.code("[" + ", ".join(f"{x:.4f}" for x in vector[:12]) + (", ...]" if len(vector) > 12 else "]"))
        st.caption("Showing the first 12 values above; the full vector is available below.")
    with col2:
        preview_count = min(40, len(vector))
        chart_df = pd.DataFrame({
            "Dimension": [f"n{i}" for i in range(preview_count)],
            "Value": vector[:preview_count]
        }).set_index("Dimension")
        st.bar_chart(chart_df)
        st.caption(f"Visualization of the first {preview_count} dimensions. Each dimension is one position in the vector.")

    with st.expander("📋 Show the complete embedding vector"):
        st.code("[" + ", ".join(f"{x:.6f}" for x in vector) + "]")
        if st.download_button(
            "⬇️ Download embedding as CSV",
            data=pd.DataFrame({"dimension": range(len(vector)), "value": vector}).to_csv(index=False),
            file_name="sentence_embedding.csv",
            mime="text/csv"
        ):
            log.info("Embedding CSV downloaded | dimensions=%d", len(vector))

    st.subheader("3️⃣ Compare meanings")
    st.write("Try two sentences. Cosine similarity gives a rough measure of how close their embeddings are.")
    c1, c2 = st.columns(2)
    with c1:
        sentence_a = st.text_input("Sentence A", value=text, key="sentence_a")
    with c2:
        sentence_b = st.text_input("Sentence B", value="I really enjoy eating mangoes", key="sentence_b")

    if st.button("🔗 Compare these sentences"):
        log.info("Compare clicked | len_a=%d | len_b=%d", len(sentence_a), len(sentence_b))
        log.debug("Sentence A: %r | Sentence B: %r", sentence_a, sentence_b)
        if not sentence_a.strip() or not sentence_b.strip():
            log.warning("Compare aborted: one or both sentences are empty")
            st.warning("Enter both sentences to compare.")
        else:
            try:
                start = time.perf_counter()
                # The model is cached, so this is instant after the first load. It must be fetched here
                # because this button click is a fresh script run where the Convert block did not execute.
                model, _ = load_model()
                emb_a, emb_b = model.encode([sentence_a, sentence_b], normalize_embeddings=True)
                score = float(cosine_similarity([emb_a], [emb_b])[0][0])
                elapsed = time.perf_counter() - start
            except Exception:
                log.exception("Failed to compare sentences")
                st.error(f"Something went wrong while comparing the sentences. Details were written to `{LOG_FILE.name}`.")
            else:
                log.info("Cosine similarity computed in %.3fs | score=%.4f", elapsed, score)
                st.metric("Cosine similarity", f"{score:.3f}")
                st.progress(max(0.0, min(1.0, (score + 1) / 2)))
                st.caption(
                    "Closer to 1 usually means more semantically similar; closer to 0 means less similar. "
                    "This is a useful intuition, not a perfect measure of meaning."
                )
                st.caption(f"Computed in {elapsed * 1000:.0f} ms · cost $0.00 (runs locally)")

    with st.expander("🧩 What just happened?"):
        st.markdown(
            """
            1. **Tokenizer:** splits the sentence into pieces.
            2. **Token IDs:** gives each piece an ID so the model can process it.
            3. **Embedding model:** uses learned numerical patterns to represent meaning and context.
            4. **Sentence embedding:** produces one fixed-size vector for the whole sentence.

            **Remember:** the token IDs are not the embedding. The final vector is the embedding you commonly store in a vector database for semantic search or RAG.
            """
        )
else:
    st.info("Enter a sentence above and click **Convert text to numbers** to explore the real output.")
    st.markdown("### Example sentences to try")
    st.write("- `I love mangoes`")
    st.write("- `I enjoy eating fruit`")
    st.write("- `The train arrives at noon`")
    st.caption("Similar ideas may have closer vectors, but the exact values depend on the model.")
