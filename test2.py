import query as q
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.document_loaders import TextLoader

def info(docs):
    print(f'You have {len(docs)} document(s)')
    num_words = sum([len(doc.page_content.split(' ')) for doc in docs])
    print(f'You have roughly {num_words} words in docs')
    print()
    print(f'Preview: \n{docs[0].page_content.split(". ")[0]}') #這行好像不是很重要

llm = OpenAI(openai_api_key="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")

query_cons = q.multiple_filter("資工系")
q.data_cleaner(query_cons)

#txt_loader
input_file = 'NTU_library.csv'  # 輸入的CSV檔案名稱
output_file = 'NTU_library.txt'  # 輸出的TXT檔案名稱
q.merge_page_content(input_file, output_file)
loader = TextLoader('NTU_library.txt')
data = loader.load()


#csv_loader
#loader = CSVLoader(file_path='NTU_library.csv', source_column='page_content')



print(info(data))


print(data)

chain = load_summarize_chain(llm, chain_type='map_reduce', verbose=True)

chain.run(data)

