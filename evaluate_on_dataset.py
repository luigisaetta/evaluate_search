"""
Process the golden dataset
"""

import json
from pathlib import Path
from typing import Iterator, Dict
from tqdm import tqdm

# using faiss as vector store
from langchain_community.vectorstores import FAISS

from oci_models import get_embedding_model
from chunk_utils import load_and_split_pdf
from config import EMBED_MODEL_URL, EMBED_MODEL_ID, CHUNK_OVERLAP, CHUNK_SIZE, K

FILE_PATTERN = "*.pdf"  # pattern to match your pdf files


def compute_recall_at_k(_true_id, _retrieved_ids):
    """
    Return 1.0 if the true_id is in the retrieved list, else 0.0.
    """
    return 1.0 if _true_id in _retrieved_ids else 0.0


def compute_reciprocal_rank(_true_id, _retrieved_ids):
    """
    Return reciprocal rank of true_id in retrieved_ids (or 0 if not found).
    """
    for rank, rid in enumerate(_retrieved_ids, start=1):
        if rid == _true_id:
            return 1.0 / rank
    return 0.0


def process_record(_record: Dict[str, str]) -> None:
    """
    Process a single JSONL record.
    The record dictionary has keys: "question", "answer", "node_id".
    Modify this function to suit your processing logic.
    """
    print(f"Question [{_record['node_id']}]: {_record['question']}")
    print(f"Answer: {_record['answer']}")
    print(f"Chunk: {_record['node_id']}")
    print("-" * 40)


def count_jsonl_records(file_path: Path) -> int:
    """
    Count the number of JSONL records in the given file.
    Each non-empty line is treated as one record (JSON object).
    """
    # Use a generator to read line by line without loading the entire file
    return sum(1 for line in file_path.open("r", encoding="utf-8") if line.strip())


def read_jsonl_lines(_file_path: Path) -> Iterator[Dict[str, str]]:
    """
    Generator that reads a JSONL file line by line.
    Yields each line as a Python dict after parsing JSON.
    """
    with _file_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                _record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decoding error in {_file_path} at line {line_number}: {e}")
                continue
            yield _record


if __name__ == "__main__":

    QA_DATASET = "./qa_dataset/dataset1.jsonl"
    # dir containing the corpus of docs
    PDF_DIR = "./input_pdf"
    input_qa_dataset = Path(QA_DATASET)

    embed_model = get_embedding_model(EMBED_MODEL_URL, EMBED_MODEL_ID)

    # we need to chunk the corpis in exactly the same way we have done when creating qa_dataset
    chunks = []
    for filepath in Path(PDF_DIR).rglob(FILE_PATTERN):
        new_docs = load_and_split_pdf(
            str(filepath), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks.extend(new_docs)

    for i, doc in enumerate(chunks):
        # chunks are numberd starting by 1
        doc.metadata["node_id"] = f"{i+1}"

    # loading in VS and embedding
    faiss_store = FAISS.from_documents(
        documents=chunks,
        embedding=embed_model,
    )
    retriever = faiss_store.as_retriever(
        search_type="similarity", search_kwargs={"k": K}
    )

    print("")
    print("Processing questions...")
    print("")

    # init stats
    total_reciprocal_rank = 0.0
    total_hit = 0.0
    n_rec = count_jsonl_records(input_qa_dataset)

    for record in tqdm(read_jsonl_lines(input_qa_dataset), total=n_rec):
        question = record["question"]
        true_id = record["node_id"]

        # here we do the search
        retrieved_docs = retriever.invoke(question)
        retrieved_ids = [doc.metadata["node_id"] for doc in retrieved_docs]

        # Compute metrics for this query
        total_reciprocal_rank += compute_reciprocal_rank(true_id, retrieved_ids)
        total_hit += compute_recall_at_k(true_id, retrieved_ids)

    # Average over all queries
    mrr_score = round(total_reciprocal_rank / n_rec, 3)
    hit_rate = round(total_hit / n_rec, 3)

    print("")
    print("Results of the test with model: ", EMBED_MODEL_ID)
    print("Hit Rate: ", hit_rate)
    print("MRR: ", mrr_score)
    print("")
