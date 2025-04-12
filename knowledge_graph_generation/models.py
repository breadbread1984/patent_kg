#!/usr/bin/python3

from os import environ
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace
from langchain_community.llms import HuggingFaceTextGenInference
import configs

class Tongyi(ChatOpenAI):
  def __init__(self,):
    super(ChatOpenAI, self).__init__(
      api_key = '',
      base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
      model_name = 'qwen2.5-7b-instruct',
      top_p = 0.8,
      temperature = 0.8,
    )

class Qwen2_5(ChatHuggingFace):
  def __init__(self,):
    environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    super(Qwen2_5, self).__init__(
      llm = HuggingFaceTextGenInference(
        inference_server_url = configs.tgi_host,
        do_sample = True,
        top_p = 0.8,
        temperature = 0.8,
      ),
      model_id = 'Qwen/Qwen2.5-7B-Instruct',
      verbose = True
    )
