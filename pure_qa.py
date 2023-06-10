from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
#from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
import query as q
#from langchain.vectorstores import FAISS

query = "資工系女生多嗎？"


llm = OpenAI(openai_api_key="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")

query_cons = q.multiple_filter(query)
q.data_cleaner(query_cons)

#txt_loader
input_file = 'NTU_library.csv'  # 輸入的CSV檔案名稱
output_file = 'NTU_library.txt'  # 輸出的TXT檔案名稱
q.merge_page_content(input_file, output_file)
loader = TextLoader('NTU_library.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

#print(documents[0])

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(llm, chain_type= "stuff")

docs = vectorstore.similarity_search(query)
a=chain.run(input_documents=docs, question=query)
print(a)
