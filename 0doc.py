import streamlit as st;
from langchain.llms import OpenAI;
from langchain.text_splitter import CharacterTextSplitter;
from langchain.embeddings import OpenAIEmbeddings;
from langchain.vectorstores import Chroma;
from langchain.chains import RetrievalQA;
import docx2txt;
import PyPDF2;
from io import BytesIO;

def generate_response(file_upload,openai_api_key,query_text):
    if file_upload is not None:
        document = file_upload.read()
        file_type =file_upload.type
        st.write("File uploaded successfully!")
        st.write("Filename:", file_upload.name)
        st.write("File type:", file_type)

    if file_type == "text/plain":
        # Process the text content of .txt file
        text = document.decode("utf-8")
        st.write("Text content:")
        st.write(text)
    
    elif file_type == "application/pdf":
        # Process the content of PDF file
        pdf_reader = PyPDF2.PdfReader(BytesIO(document))
        text =""
        for page_num in range(len(pdf_reader.numPages)):
            page = pdf_reader.getPage(page_num)
            text += page.extract_text()
        st.write("Text content:")
        st.write(text)
    
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Process the content of .docx file
        text = docx2txt.process(file_upload)
        st.write("Text content:")
        st.write(text)
        #split doc to chunks
        text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
        texts=text_splitter.create_documents(document)

        #create ebedings
        embeding=OpenAIEmbeddings(openai_api_key=openai_api_key)
        #create vectorstore
        db=Chroma.from_documents(texts,ebedings)
        #retrive 
        retiever=db.as_retriever()
        qa=RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key),chain_type='stuff',retiever=retiever)
        return qa.run(query_text)

st.set_page_config(page_title='OpenDoc')
st.title('Ask the Doc')
file_upload=st.file_uploader('upload a doc',type=["txt","docx","pdf"])
query_text=st.text_input("Enter your question ",placeholder="Ask your doc ",disabled=not file_upload)


result=[]
with st.form('myform',clear_on_submit=True):
    openai_api_key=st.text_input('OpenAi api key',type='password',disabled=not (file_upload and query_text))
    submitted=st.form_submit_button('Submit',disabled=not (file_upload and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            respones=generate_response(file_upload,openai_api_key,query_text)
            result.append(respones)
            del openai_api_key

if len(result):
    st.info(respones)
