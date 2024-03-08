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
from pdf2image import convert_from_bytes
import cv2

BODY_DETAILS_IMGS_ON_PAGE = 'Zdjęcia na stronie'

BODY_DETAILS_PAGE = "Strona"
BODY_DETAILS = "Podgląd przetwarzania"
BODY_CHAT = "Chat"
SIDEBAR_PDF_PROP = "Chat"
SIDEBAR_EXPLAIN_IMG = "Objasnienia zdjecia"
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


def body_chatbox(choice, ext_text, file_name, model):
    if choice == BODY_CHAT:
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
    result = chain({"input_documents": split_docs}, return_only_outputs=True)
    response = result["output_text"]
    return response


def setup_sidebar():
    with st.sidebar:
        st.title('🤗💬 LLM Chat App')
        st.markdown('''## Ustawienia''')
        llm = st.sidebar.selectbox('Wybierz LLM', [LLM3, LLM4])
        choice = st.radio("Co chcesz robic", [SIDEBAR_EXPLAIN_IMG, SIDEBAR_PDF_PROP], horizontal=True)
    return choice, llm


# def sidebar_chat(choice):
#     if choice == PDF_PROP:
#         display_pdf(pdf)


def sidebar_explain_img(choice):
    if choice == SIDEBAR_EXPLAIN_IMG:
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


def split_elements():
    files_path = "temp/pdf/"
    filelist = []
    content_from_each_page = {}
    screenshoot_from_each_page = []
    img_from_each_page = []
    text_from_each_page = []
    number_of_pages = 0
    for root, dirs, files in os.walk(files_path):
        for file in files:
            filelist.append(os.path.join(files_path, file))
    for files_path in filelist:
        with open(files_path, "rb") as pdf:
            sanitized_pdf_name = re.sub(r'[^A-Za-z0-9]+', '', os.path.basename(pdf.name))
            out_dir = f"temp/img/{sanitized_pdf_name}"
            create_path_if_not_exist(out_dir)
            screenshoot_from_each_page = get_page_screenshoot(out_dir, pdf, sanitized_pdf_name)
            pdf_reader = PdfReader(pdf)
            number_of_pages = len(pdf_reader.pages)
            for i, page in enumerate(pdf_reader.pages):
                text_from_each_page.append(page.extract_text())
                img_from_each_page.append(get_all_images_from_page(out_dir, i, page, sanitized_pdf_name))
    combine_data_to_single_object(content_from_each_page, img_from_each_page, screenshoot_from_each_page,
                                  text_from_each_page,
                                  number_of_pages)
    return content_from_each_page


def create_path_if_not_exist(out_dierectory):
    if not os.path.exists(out_dierectory):
        os.makedirs(out_dierectory)


def get_all_images_from_page(out_dir, page_number, page, sanitized_pdf_name):
    imgs_from_page = []
    spec_out_dir = f'{out_dir}/imgs/'
    create_path_if_not_exist(spec_out_dir)
    for i, img in enumerate(page.images):
        filename = f"{sanitized_pdf_name}_page_{page_number}_img_number_{i}.jpg"
        path_img = os.path.join(spec_out_dir, filename)
        imgs_from_page.append(path_img)
        save_image(img, path_img)
    return imgs_from_page


def save_image(img, path_img):
    with open(path_img, "wb") as fp:
        fp.write(img.data)


def combine_data_to_single_object(content, content_page_content_img, content_page_screenshot, content_page_text,
                                  number_of_pages):
    for i in range(number_of_pages):
        content[f'{i+1}'] = {
            'screenshot': content_page_screenshot[i] if i < len(content_page_screenshot) else None,
            'content_img': content_page_content_img[i] if i < len(content_page_content_img) else None,
            'text': content_page_text[i] if i < len(content_page_text) else None
        }


def get_page_screenshoot(out_dir, pdf, sanitized_pdf_name):
    page_screenshot = []
    spec_out_dir = f"{out_dir}/screenshoot/"
    create_path_if_not_exist(spec_out_dir)
    images = convert_from_bytes(pdf.read())
    for i, img in enumerate(images):
        file_name = f"{sanitized_pdf_name}_page_{i}_screenshot.jpg"
        path_img = os.path.join(spec_out_dir, file_name)
        images[i].save(path_img, 'JPEG')
        page_screenshot.append(path_img)
    return page_screenshot


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


def body_check_extracted_text(choice):
    if choice == BODY_DETAILS:
        content = split_elements()
        tab1, tab2 = st.tabs([BODY_DETAILS_PAGE, BODY_DETAILS_IMGS_ON_PAGE])
        for key, val in content.items():
            display_page_screenshot(tab1, key, val)
            display_page_imgs(tab2, key, val)


def display_page_imgs(tab, key, val):
    if not val['content_img']:
        return
    with tab:
        with st.expander(f"Strona {key}"):
            for el in val['content_img']:
                if el:
                    with st.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(el)
                        with col2:
                            st.text_area(f'Strona {key}', key=f'page_{key}_imgs_{el}', disabled=True, height=400)


def display_page_screenshot(tab, key, val):
    with tab:
        with st.expander(f"Strona {key}"):
            col1, col2 = st.columns(2)
            with col1:
                screenshot = val['screenshot']
                if screenshot:
                    st.image(screenshot)
            with col2:
                st.text_area(f'Strona {key}', key=f'page_{key}_scr', disabled=True, value=val['text'], height=300)


def clearLastSesion():
    path = "temp/"
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        except Exception as e:
            print(f"Failed to delete {item_path}. Reason: {e}")


def extract_text_from_pdf():
    path = "temp/pdf/"
    filelist = []
    for root, dirs, files in os.walk(path):
        for file in files:
            filelist.append(os.path.join(path, file))
    for path in filelist:
        with open(path, "rb") as pdf:
            text = ""
            sanitized_pdf_name = re.sub(r'[^A-Za-z0-9]+', '', os.path.basename(pdf.name))
            pdf_reader = PdfReader(pdf)
            for i, page in enumerate(pdf_reader.pages):
                text_from_page = page.extract_text()
                text += text_from_page
        return text, sanitized_pdf_name


def main():
    load_dotenv()
    sidebar_choice, model = setup_sidebar()
    st.header("Chat with PDF 💬")
    pdf = st.file_uploader("", type='pdf', accept_multiple_files=True)
    if pdf:
        save_uploadedfile(pdf)
        body_choice = st.radio("Co chcesz robic", [BODY_CHAT, BODY_DETAILS], horizontal=True)
        extracted_text, file_name = extract_text_from_pdf()
        body_chatbox(body_choice, extracted_text, file_name, model)
        body_check_extracted_text(body_choice)
        sidebar_explain_img(sidebar_choice)


def save_uploadedfile(files):
    for file in files:
        sanitized_pdf_name = re.sub(r'[^A-Za-z0-9]+', '', os.path.basename(file.name))
        with open(os.path.join('temp/pdf/', f'{sanitized_pdf_name[:-3]}.pdf'), "wb") as f:
            f.write(file.getbuffer())


if __name__ == '__main__':
    main()
