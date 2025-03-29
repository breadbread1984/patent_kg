#!/usr/bin/python3

from absl import flags, app
import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.storage import LocalFileStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from graph import get_graph
from configs import *

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('model', default = 'tongyi', enum_values = {'llama3', 'qwen2', 'gpt3.5', 'gpt4o', 'campus', 'tongyi'}, help = 'model to use')
  flags.DEFINE_integer('context_length', default = 5, help = 'content length')
  flags.DEFINE_string('chunk_dir', default = 'chunks', help = 'path to chunks')
  flags.DEFINE_string('doc_dir', default = 'docs', help = 'path to docs')

def create_interface():
  embedding = HuggingFaceEmbeddings(model_name = "intfloat/multilingual-e5-base")
  chunk_vectordb = Chroma(embedding_function = embedding, persist_directory = 'chunk_vectordb')
  chunk_store = LocalFileStore(FLAGS.chunk_dir)
  document_vectordb = Chroma(embedding_function = embedding, persist_directory = 'document_vectordb')
  doc_store = LocalFileStore(FLAGS.doc_dir)
  graph = get_graph(chunk_vectordb, chunk_store, document_vectordb, doc_store)
  def chatbot_response(user_input, history):
    chat_history = history[-2 * FLAGS.context_length:]
    chat_history.append({'role': 'user', 'content': user_input})
    history.append({'role': 'user', 'content': user_input})
    history.append({'role': 'assistant', 'content': ''})
    for event in graph.stream({'messages': chat_history}):
      if 'chatbot' in event:
        messages = event['chatbot']['messages']
        history[-1]['content'] = messages[-1].content
    return "", history
  with gr.Blocks() as demo:
    with gr.Row(equal_height = True):
      with gr.Column(scale = 15):
        gr.Markdown("<h1><center>QA System</center></h1>")
    with gr.Row():
      with gr.Column(scale = 4):
        chatbot = gr.Chatbot(type = "messages", height = 450, show_copy_button = True)
        user_input = gr.Textbox(label = '需要问什么？')
        with gr.Row():
          submit_btn = gr.Button("发送")
        with gr.Row():
          clear_btn = gr.ClearButton(components = [chatbot], value = "清空问题")
      submit_btn.click(chatbot_response,
                       inputs = [user_input, chatbot],
                       outputs = [user_input, chatbot])
  return demo

def main(unused_argv):
  demo = create_interface()
  demo.launch(server_name = service_host,
              server_port = service_port)

if __name__ == "__main__":
  add_options()
  app.run(main)
