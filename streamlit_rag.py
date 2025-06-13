import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

@st.cache_resource
def setup_rag_chain():

    loader = PyPDFLoader("docs/meme-kanseri-rehberi.pdf")
    pages = loader.load()
    all_text = "\n".join(p.page_content for p in pages)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_text(all_text)

    emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vs = Chroma.from_texts(docs, emb, persist_directory="./chroma_db")
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-preview-04-17", temperature=0.3, max_tokens=500
    )

    system_prompt = (
        "Sen bir sağlık profesyoneli gibi davranan, meme kanseri hakkında bilgi vermek için eğitilmiş bir sohbet asistanısın. "
        "Yanıtlarını yalnızca verilen bağlam içeriğinden oluştur ve PDF'te yer almayan hiçbir ek bilgi ekleme. "
        "Yanıtların bilgilendirme amaçlıdır, teşhis ve tedavi önerisi yerine geçmez. "
        "Kullanıcıdan net ve açık sorular gelmesini bekle. Sorular belirsizse, daha fazla açıklama iste. "
        "İnsanlar genellikle sorularını 'Nelerdir?', 'Nasıl yapılır?', 'Kimler risk altındadır?', 'Belirtileri nelerdir?' gibi doğal dilde sorar. "
        "Bu tür soruları anlayarak uygun başlıkları bağlamdan bulmaya çalış. "
        "Başlıklar ve içerikler tam kelime kelime uyuşmasa da, anlam olarak yakın olanları eşleştirmeye çalış. "
        "Kullanıcı teşekkür ettiğinde mutlaka \"Rica ederim, her zaman yardımcı olurum. Sağlıkla kalın.\" gibi nazik bir kapanış cümlesi ekle. "
        "Eğer sorunun cevabını bağlamda bulamazsan, \"Üzgünüm bu konuda yardımcı olamıyorum.\" de. "
        "Cevaplarını en fazla dört cümleyle ve anlaşılır bir dille yaz.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    return rag_chain

