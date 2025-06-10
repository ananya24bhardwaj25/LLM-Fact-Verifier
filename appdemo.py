import streamlit as st
from embed_store import EmbedStore
from ollama import Client
from verifier import verify_claim
st.set_page_config(page_title="LLM Fact Checker", layout="centered")
st.title("🧠 LLM Fact Checker")
st.write("Check if a claim is true based on available facts.")

claim = st.text_input("🔍 Enter a claim:")

if st.button("Verify Claim") and claim:
    with st.spinner("Retrieving facts and verifying..."):
        matched_facts = retriever.get_top_k_facts(claim, k=5)
        result = verify_claim(claim, matched_facts)

    st.subheader("🧾 Top Matching Facts:")
    for i, fact in enumerate(matched_facts, 1):
        st.write(f"**{i}.** {fact}")

    st.subheader("✅ Verdict:")
    st.success(result)
