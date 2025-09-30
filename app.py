# --- NLTK Data Download (Runs every script load) ---
logging.info("Starting NLTK data downloads (stopwords, punkt, punkt_tab, averaged_perceptron_tagger).")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)   # ✅ REQUIRED to fix your error
    nltk.download('averaged_perceptron_tagger', quiet=True)
    logging.info("NLTK data download complete.")
except Exception as e:
    logging.error(f"Error downloading NLTK data: {e}")
    st.error("Could not download necessary NLTK packages.")
