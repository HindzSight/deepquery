import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import ollama
import os
import time
import spacy

nlp = spacy.load("en_core_web_sm")

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)

# Load local text-embedding model from Hugging Face
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Create or load a Qdrant collection
collection_name = "pdf_collection"
try:
    qdrant_client.get_collection(collection_name)
except Exception as e:
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)  # 384 is the size of all-MiniLM-L6-v2 embeddings
    )

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Create directories for storing files
if not os.path.exists('files'):
    os.mkdir('files')

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text, chunk_size=1500, chunk_overlap=200):
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    
    chunks, current_chunk, current_length = [], [], 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-chunk_overlap // 10:]  # Retain overlap
            current_length = sum(len(sent.split()) for sent in current_chunk)
        current_chunk.append(sentence)
        current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def vector_store(text_chunks):
    embeddings = embedding_model.encode(text_chunks)
    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={"text": text}
        )
        for idx, (text, embedding) in enumerate(zip(text_chunks, embeddings))
    ]
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points
    )

def retrieve_relevant_chunks(question, top_k=3):
    question_embedding = embedding_model.encode(question)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=question_embedding.tolist(),
        limit=top_k
    )
    return [hit.payload["text"] for hit in search_result]

def generate_response(question, context_chunks):
    context = "\n".join(context_chunks)
    prompt = f"""You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    Context: {context}
    User: {question}
    Chatbot:"""
    
    response = ollama.generate(model="tinyllama", prompt=prompt)
    return response["response"]

def main():
    st.set_page_config("DeepQuery")
    st.title("DeepQuery")

    # Upload a PDF file
    uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

    if uploaded_file is not None:
        if not os.path.isfile("files/" + uploaded_file.name):
            with st.status("Analyzing your document..."):
                # Save the uploaded file
                bytes_data = uploaded_file.read()
                with open("files/" + uploaded_file.name, "wb") as f:
                    f.write(bytes_data)

                # Read and process the PDF
                raw_text = pdf_read([uploaded_file])
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Document processed successfully!")

        # Chat input
        if user_input := st.chat_input("You:", key="user_input"):
            user_message = {"role": "user", "message": user_input}
            st.session_state.chat_history.append(user_message)
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    # Retrieve relevant chunks and generate response
                    relevant_chunks = retrieve_relevant_chunks(user_input)
                    response = generate_response(user_input, relevant_chunks)

                # Simulate typing effect
                message_placeholder = st.empty()
                full_response = ""
                for chunk in response.split():
                    full_response += chunk + " "
                    time.sleep(0.001)
                    message_placeholder.markdown(full_response + " ")
                message_placeholder.markdown(full_response)

            chatbot_message = {"role": "assistant", "message": response}
            st.session_state.chat_history.append(chatbot_message)

    else:
        st.write("Please upload a PDF file.")

if __name__ == "__main__":
    main()