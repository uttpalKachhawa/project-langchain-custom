import os
import re
import json
import pdfplumber
import faiss
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_ollama import ChatOllama
from langchain_community.docstore.in_memory import InMemoryDocstore
from ragas.metrics import faithfulness, answer_correctness,context_precision,context_recall
from ragas import evaluate
from trulens.core import Tru
from trulens.core.instruments import instrument
from trulens.feedback import feedback
from datasets import Dataset
#from trulens.apps.langchain import 
#from trulens_eval.feedback.provider.langchain import HuggingFace



# -------------------------------
# CONFIGURATION
# -------------------------------
PDF_DIR = "pdfdoc"
FAISS_BIN = "faiss_index.bin"
METADATA_JSON = "metadata.json"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -------------------------------
# STEP 1: Extract Text from PDFs
# -------------------------------
def extract_text_tbl(pdf_path):
    """Extract text and tables from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    text += " | ".join(cell if cell else "" for cell in row)
    return text

# -------------------------------
# STEP 2: Clean and Normalize Text
# -------------------------------
def clean_text(text):
    """Clean text by removing extra spaces and page numbers."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'Page \d+', '', text)
    return text.strip()

# -------------------------------
# STEP 3: Segment Text into Chunks
# -------------------------------
def segment_text(text, chunk_size=800, chunk_overlap=100):
    """Split text into chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_text(text)

# -------------------------------
# STEP 4: Build or Load FAISS Index
# -------------------------------
# Initialize TruLens
tru = Tru()
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)


if os.path.exists(FAISS_BIN) and os.path.exists(METADATA_JSON):
    # Load FAISS index and metadata
    print("Loading FAISS index from disk...")
    index = faiss.read_index(FAISS_BIN)
    with open(METADATA_JSON, "r") as f:
        metadata_list = json.load(f)
    print(f"Loaded {len(metadata_list)} metadata entries.")
else:
    # Build FAISS index
    print("Building FAISS index...")
    documents = []
    for filename in os.listdir(PDF_DIR):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(PDF_DIR, filename))
            text = extract_text_tbl(loader.file_path)
            print(f"Text and Table : {text}")
            cleaned_text = clean_text(text)
            print(f"Cleaned Text : {clean_text}")
            chunks = segment_text(cleaned_text)
            print(f"Chunks details : {chunks}")
            for chunk in chunks:
                doc = Document(page_content=chunk, metadata={"source": filename})
                documents.append(doc)

    print(f"Total chunks created: {len(documents)}")

    # Compute embeddings
    embedding_vectors = np.array(
        [embedding_model.embed_query(doc.page_content) for doc in documents],
        dtype='float32'
    )

    # Create FAISS index
    dimension = embedding_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_vectors)

    # Save FAISS index and metadata
    faiss.write_index(index, FAISS_BIN)
    metadata_list = [doc.metadata for doc in documents]
    with open(METADATA_JSON, "w") as f:
        json.dump(metadata_list, f)
    print("FAISS index and metadata saved.")

# -------------------------------
# STEP 5: Wrap FAISS in LangChain
# -------------------------------

 #Build mappings for LangChain FAISS wrapperclass
documents = [Document(page_content="", metadata=m) for m in metadata_list]
index_to_docstore_id = {i: str(i) for i in range(len(documents))}
# Create InMemoryDocstore
docstore = InMemoryDocstore({str(i): documents[i] for i in range(len(documents))})


vectorstore = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
#retriever = vectorstore.as_retriever()
print(f"Inside retriever {retriever}")

# -------------------------------
# STEP 6: Create RAG Chain with Ollama
# -------------------------------
#inititiaize memory for chat history
#ConversationBufferWindowMemory(k=4) ensures only the last 4 interactions are kept.
#ConversationBufferdWindowMemory converting outputs into a single string
memory = ConversationBufferWindowMemory(k=4, return_messages=True,memory_key="chat_history")
llm = ChatOllama(model="phi3")
# 4. Create the ConversationalRetrievalChain
print("Creating a RAG chain with OLLAMA")
rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    output_key="answer"
)   

#initialize groundness and relevances
#f_groundedness = feedback(embedding_model.embed_documents).on_input_output()
#f_relevance = feedback(embedding_model.relevance).on_input_out

tru_rag_chain = instrument(rag_chain)
#tru_rag_chain = instrument(chain=rag_chain,
                        # app_id="RAG_with_FAISS")#,
                        # feedbacks=[f_groundedness, f_relevance])
# -------------------------------
# STEP 7: Interactive Q&A Loop
# -------------------------------
print("RAG system with TruLens ready! Type your question or 'exit' to quit.\n")

while True:
    user_question = input("Ask your question: ")
    if user_question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = rag_chain.invoke({"question": user_question,'chat_history':[]})
    answer = response["answer"]
    print("\nAnswer:", answer)
    contexts=[doc.page_content for doc in retriever._get_relevant_documents(user_question,run_manager=None)]
    # Evaluate with RAGAS
    #logs = tru.get_records_and_feedback(app_id="RAG_with_FAISS")
    eval_data={
        "question": [user_question],
        "answer": [answer],
        "context":[contexts],
        "retrieved_contexts": [contexts],
        "reference": ["Deep Bidirectional Transformers are models that process text in both directions using Transformer architecture for better context understanding."]
    }

    
    dataset= Dataset.from_dict(eval_data)
    
    result = evaluate(
    dataset,
    metrics=[faithfulness, answer_correctness, context_precision, context_recall],
    llm=llm
    )

#result=evaluate(eval_data, metrics=[faithfulness,answer_correctness,context_precision,context_recall],llm=llm,embeddings=embedding_model)
    print("\nEvaluation Metrics:", result.to_pandas)

    #ASK for feedback
    feedback= input("\nWas this answer helpful? (good/bad or comment): ")

    #upate the memory with feedback
    memory.chat_memory.add_ai_message(f"Answer: {answer} | Feedback: {feedback}")

     #optional
    if feedback.lower() == "bad":
        print("\nRegenerating answer based on feedback...")
        improved_response = rag_chain.invoke({
            "question": f"Please improve the previous answer for: {user_question}"
        })
        print("\nImproved Answer:", improved_response["answer"])
        print("-" * 80)


# Launch TruLens dashboard
#tru.run_dashboard()

#Note: Testing two things in RAGAS 
# 1. Retrieval: It belongs to more embedding model we used
# 2. Generation: It belongs to LLM used 