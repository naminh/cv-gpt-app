from langchain.llms import HuggingFaceHub
from llama_index import (PromptHelper, ServiceContext, SimpleDirectoryReader,
                         VectorStoreIndex)
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.embeddings import HuggingFaceEmbedding


class LlamaRetriever:
    def __init__(self) -> None:
        self.llm_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
        self.embedding_model_id = "BAAI/bge-small-en-v1.5"
        self.llm = self.load_llm()
        self.llama_debug = LlamaDebugHandler(print_trace_on_end=False)

    def load_llm(self):
        return HuggingFaceHub(
            repo_id=self.llm_model_id,
            model_kwargs={"temperature": 0.1, "max_new_tokens": 256},
        )

    def build_cv_qa(self, cv_path):
        callback_manager = CallbackManager([self.llama_debug])
        embed_model = HuggingFaceEmbedding(model_name=self.embedding_model_id)
        # create a service context
        service_context = ServiceContext.from_defaults(
            llm=self.llm,
            embed_model=embed_model,
            callback_manager=callback_manager,
        )

        prompt_helper = PromptHelper(
            context_window=4096,
            num_output=256,
            chunk_overlap_ratio=0.1,
            chunk_size_limit=None,
        )

        documents = SimpleDirectoryReader(input_files=[cv_path]).load_data()
        index = VectorStoreIndex.from_documents(
            documents=documents,
            service_context=service_context,
            prompt_helper=prompt_helper,
        )

        query_engine = index.as_query_engine()

        return query_engine

    def extract_last_llama_log(self):
        event_pairs = self.llama_debug.get_llm_inputs_outputs()
        return event_pairs[-1][1].payload.values()
