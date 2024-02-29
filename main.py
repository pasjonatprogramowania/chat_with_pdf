import base64
import os
import pickle
import time

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from openai import OpenAI

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

def chat_with_pdf(VectorStore, model):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
        docs = VectorStore.similarity_search(query=prompt, k=3)
        with st.chat_message("assistant"):
            chain = load_qa_chain(llm=client, chain_type="stuff")
            with get_openai_callback():
                response = chain.run(input_documents=docs, question=prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})

def setup_sidebar():
    with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        st.markdown('''## Ustawienia''')
        llm = st.sidebar.selectbox('Wybierz LLM', [LLM4, LLM3])
        choice = st.radio("Co chcesz robic", [EXPLAIN_IMG, PDF_PROP], horizontal=True)
    return choice,llm


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
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            content[f"{page}"] = page
            text_from_page = page.extract_text()
            content[f"{page}-content"] = text_from_page
            text += text_from_page
    return text, content, pdf.name if pdf else None


def split_text(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=text)
    return chunks


def get_or_create_vector_store(store_name, chunks):
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
    return VectorStore


def handle_query_and_generate_response(VectorStore,query):
    if query:
        docs = VectorStore.similarity_search(query=query, k=3)
        llm = OpenAI()
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=query)
            print(cb)
        return response


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


def sidebar_show_extracted_text(choice,text):
    if choice == PDF_PROP:
        with st.sidebar:
            col1,col2 = st.columns(2)
            with col1:
                # st.write("assafdsfdf")
                st.image("https://plus.unsplash.com/premium_photo-1675025863901-2c3fc0e28154?q=80&w=1935&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D")
            with col2:
                st.text_area('Strona 1',disabled=True,value=text,height=300)
def main():
    load_dotenv()
    choice,model = setup_sidebar()
    st.header("Chat with PDF 💬")
    pdf = st.file_uploader("", type='pdf')
    if pdf:

        file_name, extracted_text,content = body_handle_pdf(pdf)
        if extracted_text:
            chunks = split_text(extracted_text)
            VectorStore = get_or_create_vector_store(file_name[:-4], chunks)
            chat_with_pdf(VectorStore,pdf, model)
        # body_handle_text(file_name, extracted_text)
        sidebar_show_extracted_text(choice,extracted_text)
        sidebar_explain_img(choice)

def body_handle_text(store_name, text):
    if text:
        chunks = split_text(text)
        VectorStore = get_or_create_vector_store(store_name[:-4], chunks)
        handle_query_and_generate_response(VectorStore)

def body_handle_pdf(pdf):
    text,content, file_name = upload_and_extract_text(pdf)
    return file_name, text, content


if __name__ == '__main__':
    main()
