from llama_index import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.readers.faiss import FaissReader

from llama_index.llms import HuggingFaceLLM
import faiss

faiss_index = faiss.read_index("faiss_index/index.faiss")
vector_store = FaissVectorStore(faiss_index=faiss_index)

#storage_context = StorageContext.from_defaults(vector_store=vector_store)



from llama_index.prompts import PromptTemplate

# This will wrap the default prompts that are internal to llama-index
# taken from https://huggingface.co/Writer/camel-5b-hf
query_wrapper_prompt = PromptTemplate(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)

import torch

llm = HuggingFaceLLM( # too large
    context_window=2048,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.25, "do_sample": False},
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="Writer/camel-5b-hf",
    model_name="Writer/camel-5b-hf",
    device_map="auto",
    tokenizer_kwargs={"max_length": 2048},
    # uncomment this if using CUDA to reduce memory usage
    model_kwargs={"torch_dtype": torch.float16}
)

service_context = ServiceContext.from_defaults(chunk_size=512, llm=llm)

documents = SimpleDirectoryReader("./data").load_data()

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context, #storage_context=storage_context
)

query_engine = vector_store.as_query_engine()
response = query_engine.query("Who is Hamlet ?")