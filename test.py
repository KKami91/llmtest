#from dotenv import load_dotenv

#load_dotenv()


from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
#import sentence_transformers
#from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

from langchain.embeddings   import OpenAIEmbeddings

#from langchain.llms import CTransformers

import streamlit as st
import tempfile, os, datetime, requests

st.title("정신건강정보 Chat AI (PoC)")
st.write("---")

#파일 업로드

uploaded_file = st.file_uploader("정신건강정보 논문 추가")
st.write("---")


def getSTime():
    now = datetime.datetime.now()
    return now.strftime('%H:%M:%S')



# JSON 데이터를 URL에서 가져와 Chroma DB에 추가하는 함수
def add_json_from_url_to_chroma_db(url, chroma_db):
    try:
        # HTTP GET 요청으로 JSON 데이터 가져오기
        response = requests.get(url)
        response.raise_for_status()  # 오류가 발생하면 예외를 발생시킵니다.
        
        # JSON 데이터 파싱
        json_data = response.json()
        
        # Chroma DB에 JSON 데이터 추가
        chroma_db.add_data(json_data)
        
    #    print(f"Successfully added data from {url}")
    except requests.RequestException as e:
    #    print(f"Request failed: {e}")
    except Exception as e:
    #    print(f"An error occurred while adding data to Chroma DB: {e}")

# PDF 전문 파일을 Chroma DB에 추가하는 함수
def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    #print(f'{getSTime()} :Upload complete -- {uploaded_file}')
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()

    return pages

# 업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
    #print(f'{getSTime()} :PDF processing complete -- {uploaded_file}')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap=20, length_function = len, is_separator_regex= False)
    embedded_model = OpenAIEmbeddings()

    texts = text_splitter.split_documents(pages)
    #print(texts)
    db = Chroma.from_documents(texts, embedded_model)

   

    texts = text_splitter.split_documents(pages)

    db = Chroma.from_documents(texts, embedded_model)
    #print(f'{getSTime()} :Vector DB Load complete -- {db}')

    st.header("정신건강 AI에게 문의")
    question = st.text_input("문의 내용을 입력하세요")

    if st.button('문의하기'):

    #    print(f'{getSTime()} :Question -- {question}')
        llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever()) 
       
        result = qa_chain({"query": question})
    #    print(f'{getSTime()} :Retriever send complete --')
        
        st.write(result)
        
        print(result)
    #    print("----- embedding ----")
        retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
        docs = retriever_from_llm.get_relevant_documents(query=question)
        
        
    #    print(f'{getSTime()} :Result {question} --> {docs}')
   #llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
