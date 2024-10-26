import json
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.utils import truncate_text
import llama_index.core.base.response.schema 
class Rag:
    def __init__(self):
        # Load and initialize the HuggingFace embedding model
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # Set the embed model in Settings to HuggingFace
        Settings.embed_model = self.embed_model
        Settings.llm = None
        # Load JSON data only once
        with open("./data/rag/filtered_data3.json", "r", encoding="utf-8") as f:
            json_data = json.load(f)
        # Create Document objects from the data
        self.documents = [Document(text=entry['prompt'], extra_info={'response': entry['response']}) for entry in json_data]
        # Create the VectorStoreIndex using the HuggingFace embedding model
        self.index = VectorStoreIndex.from_documents(self.documents, embed_model=self.embed_model)
        self.query_engine = self.index.as_query_engine()


    def query_index(self,query):
        response = self.query_engine.query(query)
        if len(response.source_nodes)==0:
            return "No possible document for you"
        return truncate_text(self.index.docstore.docs[response.source_nodes[0].node.node_id].extra_info['response'], 1024)



if __name__ == "__main__":
    rag = Rag()
    query_string = "What are the recommended medications for Vascular lesion?"
    response = rag.query_index(query_string)
    print(f"Query Result: {response}")
# print(type(result))
