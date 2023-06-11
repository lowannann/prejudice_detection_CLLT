from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader
import query as q

query = "台大資工系的男女比例"


llm = OpenAI(openai_api_key="")

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

print(documents[0])

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(), memory=memory)

result = qa({"question": query})

print(result)