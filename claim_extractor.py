# claim_extractor.py

import spacy

class ClaimExtractor:
    def __init__(self, model="en_core_web_sm"):
        self.nlp = spacy.load(model)

    def extract_claim(self, input_text: str) -> str:
        doc = self.nlp(input_text)

        # Try to extract noun chunks and verbs to form a claim
        claims = []
        for sent in doc.sents:
            for chunk in sent.noun_chunks:
                if "government" in chunk.text.lower() or "scheme" in chunk.text.lower() or "policy" in chunk.text.lower():
                    claims.append(sent.text.strip())
                    break

        if claims:
            return claims[0]
        else:
            # fallback: return most informative sentence
            return max([sent.text for sent in doc.sents], key=len)

# Test it
if __name__ == "__main__":
    extractor = ClaimExtractor()
    input_text = "The Indian government has announced free electricity to all farmers starting July 2025."
    claim = extractor.extract_claim(input_text)
    print("🧠 Extracted Claim:", claim)
