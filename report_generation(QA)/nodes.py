#!/usr/bin/python3

from langchain_core.messages import ToolMessage
from langsmith import traceable
import tools

class BasicToolNode(object):
  """A node that runs the tools requested in the last AIMessage."""
  def __init__(self, tools: list) -> None:
    self.tools_by_name = {tool.name: tool for tool in tools}
  @traceable
  def __call__(self, inputs: dict):
    if messages := inputs.get("messages", []):
      message = messages[-1]
    else:
      raise ValueError("No message found in input")
    outputs = []
    for tool_call in message.tool_calls:
      tool_result = self.tools_by_name[tool_call["name"]].invoke(
        tool_call["args"]
      )
      outputs.append(
        ToolMessage(
          content=json.dumps(tool_result),
          name=tool_call["name"],
          tool_call_id=tool_call["id"],
        )
      )
    messages.extend(outputs)
    return {"messages": messages}
