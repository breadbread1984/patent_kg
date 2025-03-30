#!/usr/bin/python3

from langchain_openai import ChatOpenAI

class Tongyi(ChatOpenAI):
  def __init__(self,):
    super(ChatOpenAI, self).__init__(
      api_key = '',
      base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
      model_name = 'qwen2.5-7b-instruct',
      top_p = 0.8,
      temperature = 0.8,
    )

