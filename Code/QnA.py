from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
# from ibm_watsonx_ai.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
from langchain_ibm import WatsonxLLM
from pymilvus import Collection,connections
from sentence_transformers import SentenceTransformer
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app, resources={r"/ask": {"origins": "*"}})  # Enable CORS for /ask endpoint
# Load environment variables
load_dotenv()

# Connection to Milvus
def connect_to_milvus():
    connections.connect(
        alias="default",
        host=os.getenv('HOST'),
        port=os.getenv('PORT'),
        secure=bool(os.getenv('SECURE')),
        server_pem_path=os.getenv('SERVER_PEM_PATH'),
        server_name=os.getenv('SERVER_NAME'),
        user=os.getenv('USER'),
        password=os.getenv('PASSWORD')
    )
    if connections:
        print("Connected to Milvus successfully!")
        collection_name = 'better_embedding'
        col = Collection(collection_name)
        return col
    else:
        print("Failed to connect to Milvus.")
        exit(1)  # Exit the script if connection fails

# Load a pre-trained sentence transformer model
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize Watsonx LLM model
def initialize_watsonx_llm():
    api_key = os.getenv('API_KEY')
    project_id = os.getenv('PROJECT_ID')
    credentials = {
        "url": "https://us-south.ml.cloud.ibm.com",
        "apikey": api_key
    }
    model_id = ModelTypes.MIXTRAL_8X7B_INSTRUCT_V01_Q
    parameters = {
        GenParams.DECODING_METHOD: "greedy",
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 300,
        GenParams.STOP_SEQUENCES: ["/n/n","<|endoftext|>"],
        GenParams.REPETITION_PENALTY: 1.05
    }
    return WatsonxLLM(
        model_id="mistralai/mixtral-8x7b-instruct-v01",
        url=credentials.get("url"),
        apikey=credentials.get("apikey"),
        project_id=project_id,
        params=parameters
    )
all_sources = []


# Retrieve text from Milvus collection based on query vector
def retrieve_text_from_source(collection,query_vector):
    docs = []  # Initialize an empty list to store documents
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 10},
    }
    print("searching..")
    result = collection.search(query_vector, "embedding", search_params, limit=5, output_fields=["document_id","page_number","text_chunk"])
    for res in result:
        for hit in res:
            entity = hit.entity
            text_chunk = entity.get('text_chunk')
            document_id = entity.get('document_id')
            page_number = entity.get('page_number')
            doc = Document(page_content=text_chunk, metadata={'document_id': document_id, 'page_number': page_number})
            docs.append(doc)
    return docs
    
# Initialize context history list
context_history = []

# Answer a question using Watsonx LLM and Milvus embeddings
def answer_question(query, context_history,watsonx_granite, collection):
    sources=[]
    contexts=[]
    response=""
    previous_response = " "
    if context_history:
        previous_response, previous_context = context_history[-1]
    system_prompt = (
        "You are an AI assistant for answering questions about the Reserve Bank of India's monetary policies and economic updates."
        "Use only the given context to answer the questions. "
        "If user asks any followup question use the context first and then the previous reponse to answer the question"
        "If you don't know the answer, always reply with the statement Sorry, I don't know."
        # "Use three sentences maximum and keep the answer concise. "
        "Context: {context} "
        "Answer: " 
        "Previous response: {previous_response} "
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "Question: {input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm=watsonx_granite, prompt=prompt)
    query_vector = sentence_model.encode(query)
    contexts = retrieve_text_from_source(collection, [query_vector])
    print(contexts)
    sources = [f" {doc.metadata['page_number']}, {doc.metadata['document_id']}" for doc in contexts]
    response = question_answer_chain.invoke({"context": contexts, "input": query, "previous_response":previous_response})
    response_lower = response.lower()
    # print(response)
    context_history.append((response, contexts))
    if "sorry" not in response_lower :
        # print("\nReferred from:")
        for source in sources:
            # print(f"- page{source}")
            all_sources.append(source)
    return response, sources 

# Flask route for the main endpoint
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.json
        question = data.get('question')
        response, sources = answer_question(question, context_history, watsonx_granite, collection)
        response = response.replace("Assswer:", "").replace("Answer:", "").replace("AI:", "").replace("human:", "").replace("Assolution:", "").replace("Assistant:", "").strip()
        cleaned_text = re.sub(r'\b\w+:\b', '', response)
        # Remove any extra spaces that might have been left
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        return jsonify({
            'response': cleaned_text,
            'sources': sources
        })
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'response': 'Error occurred while processing the question. Please try again later.',
            'sources': []
        }), 500

# Main function
if __name__ == "__main__":
    collection = connect_to_milvus()
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    collection.create_index("embedding", index)
    print("Created index IVF_FLAT")
    collection.load()
    print("loaded")
    watsonx_granite = initialize_watsonx_llm()
    print("setup granite")
    app.run(host='0.0.0.0', port=5000, debug=False,use_reloader=False)
