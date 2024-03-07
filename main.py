import asyncio
import base64
import time
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from faiss import loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
import os
import shutil
import re
from langchain_core.documents import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

PDF_PROP = "Chat"
EXPLAIN_IMG = "Objasnienia zdjecia"
LLM4 = 'gpt-4-turbo-preview'
LLM3 = 'gpt-3.5-turbo-16k'
def response_generator():
    response = r"""Hello there! How can I assist you today?
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
                           \sum_{k=0}^{n-1} ar^k =
                           a \left(\frac{1-r^{n}}{1-r}\right)
    """

    for word in response.split():
        yield word + " "
        time.sleep(0.05)




def body_chatbox(ext_text,file_name, model):
    # client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("Co tam?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"{prompt}")
        with st.chat_message("assistant"):
            response = get_ai_response(ext_text, file_name, prompt, model)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)


def get_ai_response(ext_text, file_name, prompt, model):
    chunks = split_text(ext_text)
    db = get_or_create_vector_store(file_name, chunks)
    docs = db.similarity_search(prompt)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)
    prompt_template = f"""{prompt}:"""
    llm = ChatOpenAI(temperature=0, model_name=model)
    refine_prompt = PromptTemplate.from_template(prompt_template)

    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt2 = PromptTemplate.from_template(prompt_template)

    chain = load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=prompt2,
        verbose=True,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
    )
    result = chain({"input_documents":split_docs }, return_only_outputs=True)
    response = result["output_text"]
    return response


def setup_sidebar():
    with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        st.markdown('''## Ustawienia''')
        llm = st.sidebar.selectbox('Wybierz LLM', [LLM3, LLM4])
        choice = st.radio("Co chcesz robic", [EXPLAIN_IMG, PDF_PROP], horizontal=True)
    return choice, llm


# def sidebar_chat(choice):
#     if choice == PDF_PROP:
#         display_pdf(pdf)


def sidebar_explain_img(choice):
    if choice == EXPLAIN_IMG:
        with st.sidebar:
            st.write("Wyjaśnij zdjęcie")
            img = st.file_uploader("")
            if img:
                on = st.toggle('Pokaż podgląd')
                chosen_model = st.sidebar.selectbox('Model OCR', ['OCR', 'OpenAi-Visio'])
                if on:
                    st.image(img)
                    show_example = st.button("Pokaż")
                    if show_example:
                        st.latex(r'''
                                   a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
                                   \sum_{k=0}^{n-1} ar^k =
                                   a \left(\frac{1-r^{n}}{1-r}\right)
                                   ''')

                messages = st.container(height=300)
                if prompt := st.chat_input("Say something"):
                    messages.chat_message("user").write(prompt)
                    messages.chat_message("assistant").write(f"Echo: {prompt}")


def upload_and_extract_text(pdf):
    content = {}
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        content[f"{page}"] = page
        text_from_page = page.extract_text()
        content[f"{page}-content"] = text_from_page
        text += text_from_page
    return text, re.sub(r'[^A-Za-z0-9]+', '', pdf.name), content


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=text)
    return chunks


def get_or_create_vector_store(file_name, chunks):
    path = f"vector_stores/{file_name}"
    embeddings = OpenAIEmbeddings()
    if os.path.exists(path):
        db = FAISS.load_local(path, embeddings)
    else:
        db = FAISS.from_texts(chunks, embeddings)
        db.save_local(path)
    return db


def filter_vectore_store(embeddings):
    list_of_documents = [
        Document(page_content="foo", metadata=dict(page=1)),
        Document(page_content="bar", metadata=dict(page=1)),
        Document(page_content="foo", metadata=dict(page=2)),
        Document(page_content="barbar", metadata=dict(page=2)),
        Document(page_content="foo", metadata=dict(page=3)),
        Document(page_content="bar burr", metadata=dict(page=3)),
        Document(page_content="foo", metadata=dict(page=4)),
        Document(page_content="bar bruh", metadata=dict(page=4)),
    ]
    db = FAISS.from_documents(list_of_documents, embeddings)
    results_with_scores = db.similarity_search_with_score("foo")
    for doc, score in results_with_scores:
        print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")


def display_pdf(f):
    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    # Embedding PDF in HTML
    pdf_display = f"""<embed
    class="pdfobject"
    type="application/pdf"
    title="Embedded PDF"
    src="data:application/pdf;base64,{base64_pdf}"
    style="overflow: auto; width: 100%; height: 100vh;">"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


def sidebar_show_extracted_text(choice, text):
    if choice == PDF_PROP:
        with st.sidebar:
            col1, col2 = st.columns(2)
            with col1:
                # st.write("assafdsfdf")
                st.image(
                    "https://plus.unsplash.com/premium_photo-1675025863901-2c3fc0e28154?q=80&w=1935&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
            with col2:
                st.text_area('Strona 1', disabled=True, value=text, height=300)


def clearLastSesion():
    path = "vector_store"
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


def main():
    load_dotenv()
    # clearLastSesion()
    choice, model = setup_sidebar()
    st.header("Chat with PDF 💬")
    pdf = st.file_uploader("", type='pdf')
    if pdf:
        extracted_text, file_name, content = upload_and_extract_text(pdf)
        body_chatbox(extracted_text,file_name, model)
        sidebar_show_extracted_text(choice, extracted_text)
        sidebar_explain_img(choice)


if __name__ == '__main__':
    main()
