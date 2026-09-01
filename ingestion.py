import asyncio
import os
import ssl
from typing import Any, Dict, List

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

from logger import Colors, log_error, log_header, log_info, log_success, log_warning

load_dotenv()

# Configure SSL context to use a certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()


# Create an OpenAI embedding client using text-embedding-3-small; don't display a progress bar; batch embedding inputs in groups of up to 50; and cap retry backoff delays at 10 seconds.
# Create several document batches of 50 and send them (50 at a time) to get vectorized
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_max_seconds=10,
)

# chroma = Chroma(persist_directory="chroma_df",embedding=embeddings)
vectorstore = PineconeVectorStore(
    index_name="langchain-docs-index", embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


async def main():
    """Main async function to orchestrate the entire process"""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "TavilyCrawl: Starting to Crawl Documentation from https://python.langchain.com",
        Colors.PURPLE,
    )

    # Crawl the documentation

    # res = tavily_crawl.invoke(
    #     {
    #         "url": "https://docs.langchain.com/oss/python/langchain",
    #         "max_depth": 5,
    #         "extra_depth": "advanced",
    #         # "instructions": "content on ai agents",
    #     }
    # )

    res = tavily_crawl.invoke(
        {
            "url": "https://docs.langchain.com/oss/python/langchain",
            "max_depth": 5,
            "max_breadth": 50,
            "max_pages": 1000,
            "extract_depth": "advanced",
        }
    )

    all_docs = [
        Document(page_content=result["raw_content"], metadata={"source": result["url"]})
        for result in res["results"]
        if result.get("raw_content")
    ]

    log_success(
        f"TavilyCrawl: Successfuly crawled {len(all_docs)} URLs from documentation site"
    )

    # Create Batches

    async def index_documents_async(documents: List[Document], batch_size: int = 50):
        """Process documents in batches asynchronously."""
        log_header("VECTOR STORAGE PHASE")
        log_info(
            f"VectoStore:Indexing: Preparting to add {len(documents)} documents to vectore store",
            Colors.DARKCYAN,
        )

        # Create Batches

        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        log_info(
            f"VectorStore Indexing: Split into {len(batches)} batches of {batch_size} docuements each"
        )

        # Process all batches concurrently
        async def add_batch(batch: List[Document], batch_num: int):
            try:
                await vectorstore.aadd_documents(batch)
                log_success(
                    f"VectorStore INdexing: Sucessfully added batch {batch_num}/{len(batches)} ({len(batch)} documents)"
                )
            except Exception as e:
                log_error(f"VectorStore Indexing: Failed to add bat {batch_num}-{e}")
                return False
            return True

        tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Count sucessful batches

        successful = sum(1 for result in results if result is True)

        if successful == len(batches):
            log_success(
                f"VectorStore Indexing: All batches process sucessfuly! {successful}/{len(batches)}"
            )
        else:
            log_warning(
                f"VectorStore Indexing: Processed {successful}/{len(batches)} batches successfully"
            )

    # Split documents into chucks
    log_header("DOCUMENT CHUNKING PHASE")

    log_info(
        f"Text Splitter: Processing {len(all_docs)} documents "
        f"with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)

    splitted_docs = text_splitter.split_documents(all_docs)

    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks "
        f"from {len(all_docs)} documents"
    )

    await index_documents_async(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    log_success("🎉 Documentation ingestion pipeline finished successfully!")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"       URLs mapped: {len(res['results'])}")
    log_info(f"       Documents extracted: {len(all_docs)}")
    log_info(f"       Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
