import cohere
from langchain_community.retrievers.wikipedia import WikipediaRetriever
from pydantic import BaseModel
from typing import List

# Set up Cohere API key
cohere_api_key = 'sRmFY97EVTJa7VaaaQha5oH7lScl1rxTZv8x6KrV'
co = cohere.Client(cohere_api_key)

# Initialize Wikipedia Retriever
getti = WikipediaRetriever(lang='en', doc_content_chars_max=500000, top_k_results=5, sleep_time=0, max_retry=12)

class Document(BaseModel):
    page_content: str
    metadata: dict

def combine_documents_fn(docs: List[Document]) -> str:
    document_context = ""
    for doc in docs:
        document_context += doc.page_content
    return document_context

def retrieve_docs(question: str) -> List[Document]:
    documents = getti.get_relevant_documents(question)
    return documents

def generate_answer(context, question):
    prompt_template = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    response = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt_template,
        max_tokens=150
    )
    return response.generations[0].text.strip()

def ask_question(question: str):
    try:
        docs = retrieve_docs(question)
        context = combine_documents_fn(docs)
        answer = generate_answer(context, question)
        return answer
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    question = input("Ask your question: ")
    answer = ask_question(question)
    print("Answer:", answer)
