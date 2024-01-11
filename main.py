

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

embedding_model_id = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model_id,
)

def generate_embeddings():
    chunk_size = 2048

    with open("data/shakespeare.txt") as f:
        text = f.read()

        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
        pages = text_splitter.split_text(text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
        texts = text_splitter.create_documents(pages)

    print(len(texts))



    embeddings_db = FAISS.from_documents(texts, embeddings)
    embeddings_db.save_local("faiss_index")

print("start load embeddings")
embeddings_db = FAISS.load_local("faiss_index", embeddings)
print("finish load embeddings")

retriever = embeddings_db.as_retriever(search_kwargs={"k": 10})


### prompt generation
 # Define prompt template

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoTokenizer

model_name = "Intel/dynamic_tinybert"

# Load the tokenizer associated with the specified model
tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, max_length=512)

question_answerer = pipeline(
    "question-answering",
    model=model_name,
    tokenizer=tokenizer,
    return_tensors='pt'
)

# Create an instance of the HuggingFacePipeline, which wraps the question-answering pipeline
# with additional model-specific arguments (temperature and max_length)
llm = HuggingFacePipeline(
    pipeline=question_answerer,
    model_kwargs={"temperature": 0.7, "max_length": 50},
)


prompt_template = """
As literature critic answer me

question: {question}

context: {context}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])


chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs = {"prompt": prompt})

question = "Who is Hamlet ?"
answer = chain.invoke({"query": question})
print(answer)