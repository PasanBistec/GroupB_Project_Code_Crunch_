from langchain.document_loaders import YoutubeLoader as CustomYoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter as CustomTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings as CustomOpenAIEmbeddings
from langchain.vectorstores import FAISS as CustomFAISS
from langchain.llms import OpenAI as CustomOpenAI
from langchain import PromptTemplate as CustomPromptTemplate
from langchain.chains import LLMChain as CustomLLMChain
from dotenv import load_dotenv

load_dotenv()

embeddings = CustomOpenAIEmbeddings()

def create_custom_db_from_youtube_video_url(video_url: str) -> CustomFAISS:
    custom_loader = CustomYoutubeLoader.from_youtube_url(video_url)
    transcript = custom_loader.load()
    text_splitter = CustomTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    custom_db = CustomFAISS.from_documents(docs, embeddings)
    return custom_db

def get_custom_response_from_query(db, query, k=4):
    custom_docs = db.similarity_search(query, k=k)
    custom_docs_page_content = " ".join([d.page_content for d in custom_docs])
    custom_llm = CustomOpenAI(model_name="text-davinci-003")
    custom_prompt = CustomPromptTemplate(
        input_variables=["question", "docs"],
        template="""
        You are a helpful assistant that can answer questions about YouTube videos 
        based on the video's transcript.
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        Only use the factual information from the transcript to answer the question.
        If you feel like you don't have enough information to answer the question, say "I don't know".
        Your answers should be verbose and detailed.
        """,
    )
    custom_chain = CustomLLMChain(llm=custom_llm, prompt=custom_prompt)
    custom_response = custom_chain.run(question=query, docs=custom_docs_page_content)
    custom_response = custom_response.replace("\n", "")
    return custom_response, custom_docs