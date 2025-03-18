#!/usr/bin/python3

# huggingface_token = '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'
huggingface_token = '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'

tgi_host = "http://localhost:9090"

neo4j_host = "bolt://localhost:7687"
neo4j_user = "neo4j"
neo4j_password = "12345678"
neo4j_db = "neo4j"

use_fewshot = False
use_selector = False

node_types = ['Patent', 'Inventor', 'Applicant', 'Technical Field', 'Patent Owner', 'Patent Citation']

rel_types = [
  ('Patent', 'Invented_by', 'Inventor'),
  ('Patent', 'Applied_by', 'Applicant'),
  ('Patent', 'Belongs_to', 'Technical Field'),
  ('Patent', 'Owned_by', 'Patent Owner'),
  ('Patent', 'Cite', 'Patent Citation')
]

examples = [{
        "text": "The present invention provides an apparatus for facilitating a photovoltaic device to provide a wireless communication channel. The apparatus comprises a switch connected in parallel with the photovoltaic device and configured for driving the photovoltaic device to produce optical signals carrying sensed data to be transmitted; and a control module connected with the switch and configured for receiving electrical sensing signals and generate a control signal to control the switch.",
        "extracted_info": {
            "head": "Apparatus for facilitating a photovoltaic device to provide a wireless communication channel",
            "head_type": "Patent",
            "relation": "Belongs_to",
            "tail": "Photovoltaic devices",
            "tail_type": "Technical Field"
        }
    },
    {
        "text": "Inventor(s): Garaj; Martin (Bratislava, SK), Chung; Shu Hung Henry (Hong Kong, HK)",
        "extracted_info": {
            "head": "11671731",
            "head_type": "Patent",
            "relation": "Invented_by",
            "tail": "Martin Garaj",
            "tail_type": "Inventor"
        }
    },
    {
        "text": "Applicant: City University of Hong Kong (Hong Kong, HK)",
        "extracted_info": {
            "head": "11671731",
            "head_type": "Patent",
            "relation": "Applied_by",
            "tail": "City University of Hong Kong",
            "tail_type": "Applicant"
        }
    },
    {
        "text": "US Patent 9413457 is cited in this invention as prior art.",
        "extracted_info": {
            "head": "11671731",
            "head_type": "Patent",
            "relation": "Cite",
            "tail": "9413457",
            "tail_type": "Patent Citation"
        }
    }
]
