from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from llm_agent import ask_llm


def load_pdf_chunks(file, name, size=800, overlap=150):
    chunks = []
    for page_no, page in enumerate(PdfReader(file).pages, start=1):
        text = (page.extract_text() or "").strip()
        i = 0
        while i < len(text):
            chunks.append({"doc": name, "page": page_no, "text": text[i:i + size]})
            i += size - overlap
    if not chunks:
        raise ValueError("No text could be extracted from the PDF (it may be a scanned document).")
    return chunks


class SimpleRAG:
    def __init__(self, chunks):
        self.chunks = chunks
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = self.vec.fit_transform([c["text"] for c in chunks])

    def search(self, question, k=3):
        scores = cosine_similarity(self.vec.transform([question]), self.matrix)[0]
        top = scores.argsort()[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def answer(self, question):
        hits = self.search(question)
        if not hits or hits[0][1] < 0.05:
            return "The document does not contain supporting information for this question.", []
        context = "\n\n".join(f"[{c['doc']} p.{c['page']}] {c['text']}" for c, _ in hits)
        prompt = f"""Answer ONLY from the context below. If the answer is not there, say you could not find it.
Context:
{context}

Question: {question}
Answer in clear, professional English only, even if the question is written in Hindi or Hinglish. Do not use Hindi or Hinglish words."""
        return ask_llm(prompt), sorted({(c["doc"], c["page"]) for c, _ in hits})
