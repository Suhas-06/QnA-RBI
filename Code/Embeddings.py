import os
import re
import np
from dotenv import load_dotenv
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from pymilvus import Collection,connections,DataType,FieldSchema,CollectionSchema

load_dotenv()
# Connection to Milvus
def connect_to_milvus():
    connections.connect(
        alias="default",
        host=os.getenv('HOST'),
        port=os.getenv('PORT'),
        secure="bool(os.getenv('SECURE'))",
        server_pem_path=os.getenv('SERVER_PEM_PATH'),
        server_name=os.getenv('SERVER_NAME'),
        user=os.getenv('USER'),
        password=os.getenv('PASSWORD')
    )

model = SentenceTransformer('all-MiniLM-L6-v2') #smaller and more efficient

# Preprocess text
def preprocess_text(text):
    """
    Lowercase text and remove non-alphanumeric characters.
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s.]', '', text)
    tokens = text.split()
    return " ".join(tokens)

# Extract text from a single PDF file
def extract_text_from_pdf(pdf_path):
    text = []
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text.append((page_num, page.get_text()))
    return text

# Generate and print embeddings for a single PDF
def generate_and_store_embeddings(pdf_path, document_id):
    """
    Generate embeddings for text chunks extracted from a PDF and insert into Milvus collection.
    """
    connect_to_milvus()
    pages = extract_text_from_pdf(pdf_path)
    entities = []

    for page_num, text in pages:
        processed_text = preprocess_text(text)
        chunks = [processed_text[i:i+512] for i in range(0, len(processed_text), 512)]
        embeddings = [model.encode(chunk) for chunk in chunks]
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            embedding_np = np.array(embedding)  # Convert embedding to numpy array
            entities.append({
                "document_id": document_id,
                "page_number": page_num,
                "text_chunk": chunk,
                "embedding": embedding_np.tolist()  # Convert to list for Milvus storage
            })

    max_chunk_length = max(len(entity["text_chunk"]) for entity in entities)
    collection_name = 'better_embedding'
    max_length_text_chunk= max_chunk_length + 128
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR,max_length=256),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="text_chunk", dtype=DataType.VARCHAR,max_length=max_length_text_chunk),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields)
    embeddings = Collection("better_embedding", schema, consistency_level="Strong")
    collection = Collection(collection_name)
    result = collection.insert(entities)
    # Check if insertion was successful
    if result.primary_keys:
        print(f"Added {len(result.primary_keys)} rows/entities to the collection.")
    

# Process all PDF files in a given directory
def process_pdfs_in_directory(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(directory_path, filename)
            print(f"Processing {pdf_path}...")
            generate_and_store_embeddings(pdf_path,document_id=filename)

# Main function
if __name__ == "__main__":
    directory_path = "PATH_TO_YOUR_DIRECTORY"
    process_pdfs_in_directory(directory_path)
