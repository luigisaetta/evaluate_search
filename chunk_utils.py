"""
chunk utils
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils import remove_path_from_ref, get_console_logger
from config import CHUNK_SIZE, CHUNK_OVERLAP

logger = get_console_logger()


def get_chunk_header(file_path):
    """
    Generate an header for the chunk.

    For now, contains only the pdf title
    """
    doc_name = remove_path_from_ref(file_path)
    # split to remove the extension
    doc_title = doc_name.split(".")[0]

    return f"# Doc. title: {doc_title}\n", doc_name


def get_recursive_text_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Return a Langchain recursive Text Splitter
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter


def load_and_split_pdf(book_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Loads and splits a PDF document into chunks using a recursive character text splitter.

    Args:
        book_path (str): The file path of the PDF document.
        chunk_size (int): Size of each text chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        List[Document]: A list of LangChain Document objects with metadata.
    """
    text_splitter = get_recursive_text_splitter(chunk_size, chunk_overlap)

    loader = PyPDFLoader(file_path=book_path)

    docs = loader.load_and_split(text_splitter=text_splitter)

    chunk_header = ""

    if len(docs) > 0:
        chunk_header, _ = get_chunk_header(book_path)

    # remove path from source and reduce the metadata (16/03/2025)
    for doc in docs:
        # add more context to the chunk
        doc.page_content = chunk_header + doc.page_content
        doc.metadata = {
            "source": book_path,
            "page_label": doc.metadata.get("page_label", ""),
        }

    logger.info("Successfully loaded and split %d chunks from %s", len(docs), book_path)

    return docs
