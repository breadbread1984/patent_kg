#!/usr/bin/python3

from os import walk
from os.path import splitext, join, exists
from absl import flags, app
from tqdm import tqdm
import json
from transformers import AutoTokenizer
from langchain.document_loaders import UnstructuredPDFLoader, UnstructuredHTMLLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_neo4j import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer, DiffbotGraphTransformer, RelikGraphTransformer, GlinerGraphTransformer
from configs import *
from models import Tongyi

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('input_dir', default = None, help = 'path to directory')
  flags.DEFINE_boolean('split', default = False, help = 'whether to split document')
  flags.DEFINE_enum('model', default = 'llm', enum_values = {'llm', 'diffbot', 'relik', 'gliner'}, help = 'model type')

def main(unused_argv):
  llm = Tongyi()
  if FLAGS.model == 'llm':
    graph_transformer = LLMGraphTransformer(llm = llm,)
  elif FLAGS.model == 'diffbot':
    graph_transformer = DiffbotGraphTransformer(diffbot_api_key = diffbot_api_key)
  elif FLAGS.model == 'relik':
    graph_transformer = RelikGraphTransformer()
  elif FLAGS.model == 'gliner':
    graph_transformer = GlinerGraphTransformer()
  else:
    raise Exception('unknown graph transformer type!')
  neo4j = Neo4jGraph(url = neo4j_host, username = neo4j_user, password = neo4j_password, database = neo4j_db)
  if FLAGS.split:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
  for root, dirs, files in tqdm(walk(FLAGS.input_dir)):
    for f in files:
      stem, ext = splitext(f)
      if ext.lower() in ['.htm', '.html']:
        loader = UnstructuredHTMLLoader(join(root, f))
      elif ext.lower() == '.txt':
        loader = TextLoader(join(root, f))
      elif ext.lower() == '.pdf':
        loader = UnstructuredPDFLoader(join(root, f), mode = 'single')
      else:
        raise Exception('unknown format!')
      docs = loader.load()
      if FLAGS.split:
        docs = text_splitter.split_documents(docs)
      graph = graph_transformer.convert_to_graph_documents(docs)
      neo4j.add_graph_documents(graph)

if __name__ == "__main__":
  add_options()
  app.run(main)
