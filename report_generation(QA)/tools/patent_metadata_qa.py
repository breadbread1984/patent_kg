#!/usr/bin/python3

from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain_core.runnables import Runnable
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools import StructuredTool
from langchain_neo4j import Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from configs import *

def load_patent_metadata_qa(llm):
  class PatentMetadataQAInput(BaseModel):
    query: str = Field(description = 'question about patent metadata')
  class PatentMetadataQAOutput(BaseModel):
    response: str = Field(description = 'answer to the question')
  class PatentMetadataQAConfig(BaseModel):
    class Config:
      arbitrary_types_allowed = True
    chain: Runnable
  class PatentMetadataQATool(StructuredTool):
    name: str = "patent_metadata_query"
    description: str = "tool for answering question about patent metadata"
    args_schema: Type[BaseModel] = PatentMetadataQAInput
    config: PatentMetadataQAConfig
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> PatentMetadataQAOutput:
      response = self.config.chain.invoke(query)
      return PatentMetadataQAOutput(response = response['result'])
  graph = Neo4jGraph(url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db)
  graph.refresh_schema()
  chain = GraphCypherQAChain.from_llm(
    graph = graph,
    llm = llm,
    verbose = True,
    allow_dangerous_requests = True
  )
  return PatentMetadataQATool(config = PatentMetadataQAConfig(chain = chain))

