import sys

import gradio as gr
from loguru import logger

from src.llamaindex_utils import LlamaRetriever
from src.utils import decrypt_cv

logger.remove(0)  # remove the default handler configuration
logger.add(
    sys.stdout, level="INFO", serialize=True, format="{time} - {level} - {message}"
)

cv_decrypt_path = decrypt_cv("cv.pdf")
llama_retriever = LlamaRetriever()
qa_engine = llama_retriever.build_cv_qa(cv_path=cv_decrypt_path)

with gr.Blocks() as app:

    def predict(message, history):
        response = qa_engine.query(message)
        logger.info(llama_retriever.extract_last_llama_log())
        return response.response

    gr.ChatInterface(
        predict,
        title="LLM Chat with Minh's CV",
        description="""Chat application to ask questions about my CV.
        
        DEMO app only - for more accurate answers, reach out to me personally for discussions. :)
        """,
        theme=gr.themes.Monochrome(),
        # clear_btn=None,
    )

    # reset_btn = gr.Button("Reset Chat")

    # reset_btn.click(fn=clear_chat_memory)
    # @reset_btn.click()
    # def clear_chat_memory():
    #     chat_memory.clear()


app.launch()
