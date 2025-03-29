#!/usr/bin/python3

from os import environ
import uuid
from langchain.agents import load_tools, AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActJsonSingleInputOutputParser
from langchain.tools.render import render_text_description
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_neo4j import Neo4jVector
from langsmith import traceable, Client
from models import Llama3_2, Qwen2_5, GPT35Turbo, GPT4O, Campus, Tongyi
from tools import load_chunk_retriever, load_document_retriever, load_patent_metadata_qa
from prompts import react_prompt
from configs import *

class Agent(object):
  def __init__(self, model, chunk_vectordb, chunk_store, doc_vectordb, doc_store):
    environ['LANGSMITH_TRACING'] = langsmith_trace
    environ['LANGSMITH_API_KEY'] = langsmith_api_key
    environ['LANGSMITH_PROJECT'] = 'patent report'
    self.client = Client()
    self.client.create_feedback(str(uuid.uuid4()), key = 'user-score', score = 1.0)
    llms_types = {
      'llama3': Llama3_2,
      'qwen2': Qwen2_5,
      'gpt3.5': GPT35Turbo,
      'gpt4o': GPT4O,
      'campus': Campus,
      'tongyi': Tongyi
    }
    llm = llms_types[model]()
    tools = [load_chunk_retriever(chunk_vectordb, chunk_store),
             load_document_retriever(doc_vectordb, doc_store),
             load_patent_metadata_qa(llm),]
    prompt = react_prompt
    # adapt prompt to openai's preference
    if model in ['gpt3.5', 'gpt4o']:
      for i in range(len(prompt)):
        if prompt[i][0] == 'user':
          prompt[i] = ('human', prompt[i][1])
    prompt = prompt.partial(
      tools = render_text_description(tools),
      tool_names = ", ".join([t.name for t in tools])
    )
    chain = {
      "input": lambda x: x["input"],
      "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
      "chat_history": lambda x: x["chat_history"]
    } | prompt | llm | ReActJsonSingleInputOutputParser()
    self.agent_chain = AgentExecutor(agent = chain, tools = tools, verbose = True, handle_parsing_errors = True)
  @traceable
  def query(self, question, chat_history):
    return self.agent_chain.invoke({"input": question, "chat_history": chat_history})
