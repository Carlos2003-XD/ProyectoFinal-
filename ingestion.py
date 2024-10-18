import os

from dotenv import load_dotenv
load_dotenv()

from consts import INDEX_NAME
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from firecrawl import FirecrawlApp
from langchain.schema import Document

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
def ingest_docs():
    loader = ReadTheDocsLoader("./langchain-docs/api.python.langchain.com/en/latest/", encoding='UTF-8')
    raw_documents = loader.load()
    print(f"loaded{len(raw_documents)} raw documents")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    documents = text_splitter.split_documents(raw_documents)
    print(f"loaded {len(documents)} documents")
    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")
    PineconeVectorStore.from_documents(
        documents, embeddings, index_name=INDEX_NAME
    )




def ingest_docs2()-> None:
    app = FirecrawlApp(api_key=os.environ['FIRECRAWL_API_KEY'])

    url = "https://marvel.fandom.com/es/wiki/Ghost_Rider_(película), https://www.zonanegativa.com/ghost-rider-el-motorista-fantasma-de-mark-steven-johnson/, https://www.almasoscuras.com/art/1929/ghost_rider_2_espiritu_de_venganza, https://www.fotogramas.es/peliculas-criticas/a468204/ghost-rider-espiritu-de-venganza/, https://comicritico.blogspot.com/2014/10/ghost-rider-2007.html"

    page_content = app.scrape_url(url=url,
                                  params={
                                      "onlyMainContent": True
                                  })
    print(page_content)
    doc = Document(page_content=str(page_content), metadata={"source": url})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents([doc])

    PineconeVectorStore.from_documents(
        docs, embeddings, index_name="firecrawl"
    )


if __name__ == "__main__":
    ingest_docs2()