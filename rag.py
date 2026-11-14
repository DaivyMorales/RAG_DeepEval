import os

from dotenv import load_dotenv
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

load_dotenv()

STORAGE_DIR = "storage"


Settings.llm = OpenAILike(model="gpt-6-luna", is_chat_model=True, context_window=128000)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

# Build the index once and save it; later runs load it from disk instead of re-embedding.
# Delete the storage/ folder after changing files in data/ or the embedding model.
if os.path.exists(STORAGE_DIR):
    storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    index = load_index_from_storage(storage_context)
else:
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=STORAGE_DIR)

query_engine = index.as_query_engine()

if __name__ == "__main__":
    while True:
        question = input("\nPregunta (o 'salir'): ").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            break
        print(query_engine.query(question))
