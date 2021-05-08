import pandas as pd
import streamlit as st
from keybert import KeyBERT
from samples import texts


@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=True)
def load_model():
    model = KeyBERT("distilbert-base-nli-mean-tokens")
    return model


model = load_model()


placeholder = st.empty()
text_input = placeholder.text_area("Type in some text you want to analyze", height=300)

sample_text = st.selectbox(
    "Or pick some sample texts", [f"sample {i+1}" for i in range(len(texts))]
)

sample_id = int(sample_text.split(" ")[-1])
text_input = placeholder.text_area(
    "Type in some text you want to analyze", value=texts[sample_id - 1], height=300
)


top_n = st.sidebar.slider("Select number of keywords to extract", 5, 20, 10, 1)
min_ngram = st.sidebar.number_input("Min ngram", 1, 5, 1, 1)
max_ngram = st.sidebar.number_input("Max ngram", min_ngram, 5, 3, step=1)
st.sidebar.code(f"ngram_range = ({min_ngram}, {max_ngram})")


params = {
    "docs": text_input,
    "top_n": top_n,
    "keyphrase_ngram_range": (min_ngram, max_ngram),
    "stop_words": "english",
}

add_diversity = st.sidebar.checkbox("Add diversity to the results")

if add_diversity:
    method = st.sidebar.selectbox(
        "Select a method", ("Max Sum Similarity", "Maximal Marginal Relevance")
    )
    if method == "Max Sum Similarity":
        nr_candidates = st.sidebar.slider("nr_candidates", 20, 50, 20, 2)
        params["use_maxsum"] = True
        params["nr_candidates"] = nr_candidates

    elif method == "Maximal Marginal Relevance":
        diversity = st.sidebar.slider("diversity", 0.1, 1.0, 0.6, 0.01)
        params["use_mmr"] = True
        params["diversity"] = diversity


keywords = model.extract_keywords(**params)

if keywords != []:
    st.info("Extracted keywords")
    keywords = pd.DataFrame(keywords, columns=["keyword", "relevance"])
    st.table(keywords)
