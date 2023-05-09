import os
from langchain.document_loaders import PyPDFLoader 
from langchain.embeddings import OpenAIEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

from dotenv import load_dotenv   #for python-dotenv method
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.environ.get('OPENAI-KEY')

pdf_path = "india.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()

embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(pages, embedding=embeddings, 
                                 persist_directory=".")
vectordb.persist()

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.9) , 
                                               vectordb.as_retriever(), memory=memory)

query = "who represented constitution?"
result = pdf_qa({"question": query})
print("Answer:")
print(result["answer"])
