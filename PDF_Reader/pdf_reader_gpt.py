# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 17:33:09 2023

@author: Nicholas Jernigan (nicholasjernigan@gmail.com)
"""
import os
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI



os.environ["OPENAI_API_KEY"] = "your-gpt-key"
path = 'Happyness Secret.pdf'          
query = "What is the key to happiness according to this article?"



   
#Prep gpt stuff
reader = PdfReader(path)
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
chain = load_qa_chain(OpenAI(), chain_type="stuff")



#Step one, find invoice number and check to see if already in sheet

docs = docsearch.similarity_search(query)
gpt_response = chain.run(input_documents=docs, question=query)               
print(gpt_response)