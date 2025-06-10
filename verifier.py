from ollama import Client
from embed_store import EmbedStore

client = Client(host='http://localhost:11434')

def verify_claim(claim, facts):
    context = "\n".join([f"- {fact}" for fact in facts[:5]])

    prompt = f"""
You are a fact-checking assistant. Use the provided facts to verify the claim.

Claim: "{claim}"

Facts:
{context}

Respond with one of the following:
- TRUE: if the claim matches the facts
- FALSE: if the claim contradicts the facts
- UNVERIFIABLE: if the facts do not provide enough information

Also, give a short reason.
"""

    response = client.chat(
        model="mistral",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for verifying factual claims."},
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"].strip()


# ✅ Load index once at the top
retriever = EmbedStore()
retriever.load_index()

# ✅ Continuous input loop
while True:
    claim = input("\nEnter a claim to verify (or type 'exit'): ")
    if claim.lower() == 'exit':
        break

    matched_facts = retriever.get_top_k_facts(claim, k=5)
    result = verify_claim(claim, matched_facts)

    print("\n🔍 Top Matching Facts:")
    for fact in matched_facts:
        print("-", fact)

    print(f"\n✅ Model's Verdict:\n{result}")
