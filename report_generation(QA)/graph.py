#!/usr/bin/python3

from os import environ
import uuid
from typing import Annotated, TypedDict, List, Dict
from langgraph.graph import StateGraph, START, END
from langsmith import traceable, Client
from models import Tongyi
from tools import load_chunk_retriever, load_document_retriever, load_patent_metadata_qa
from nodes import BasicToolNode
from configs import *

class State(TypedDict):
  messages: list

def get_graph(chunk_vectordb, chunk_store, document_vectordb, doc_store):
  environ['LANGSMITH_TRACING'] = langsmith_trace
  environ['LANGSMITH_API_KEY'] = langsmith_api_key
  environ['LANGSMITH_PROJECT'] = 'patent report'
  client = Client()
  client.create_feedback(str(uuid.uuid4()), key = 'user-score', score = 1.0)
  graph_builder = StateGraph(State)
  llm = Tongyi()
  tools = [load_chunk_retriever(chunk_vectordb, chunk_store),
           load_document_retriever(doc_vectordb, doc_store),
           load_patent_metadata_qa(llm),]
  @traceable
  def assemble_prompt(state: State):
    messages = state['messages']
    messages.insert(0, {'role': 'system', 'content': """You are a report generation assistant tasked with producing a well-formatted context given parsed context.
You will be given context from one or more reports that take the form of parsed text.
You are responsible for producing a report with text.
You have access to the following tools:\n\n{tools}\n\nThe way you use the tools is by specifying a json blob.\nSpecifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).\n\nThe only values that should be in the "action" field are: {tool_names}\n\nThe $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:\n\n```\n{{\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n```\n\nALWAYS use the following format:\n\nQuestion: the input question you must answer\nThought: you should always think about what to do\nAction:\n```\n$JSON_BLOB\n```\nObservation: the result of the action\n... (this Thought/Action/Observation can repeat N times)\nThought: I now know the final answer\nFinal Answer: the final answer to the original input question\n\nBegin! Reminder to always use the exact characters `Final Answer` when responding."""})
    return {'messages': messages}
  graph_builder.add_node('prompt', assemble_prompt)
  llm_with_tools = llm.bind_tools(tools)
  @traceable
  def chatbot(state: State):
    messages = state['messages']
    messages.append(llm_with_tools.invoke(messages))
    return {"messages": messages}
  graph_builder.add_node("chatbot", chatbot)
  @traceable
  def route_tools(state: State):
    if isinstance(state, list):
      ai_message = state[-1]
    elif messages := state.get('messages', []):
      ai_message = messages[-1]
    else:
      raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
      return "tools"
    return END
  tool_node = BasicToolNode(tools = tools)
  graph_builder.add_node("tools", tool_node)
  # define edges
  graph_builder.add_edge(START, "prompt")
  graph_builder.add_edge("prompt", "chatbot")
  graph_builder.add_conditional_edges("chatbot", route_tools, {"tools": "tools", END: END})
  graph_builder.add_edge("tools", "chatbot")
  graph = graph_builder.compile()
  return graph

