"""
This script generates a golden Q&A dataset using an LLM with LangChain Expression Language (LCEL).
It reads documents from a directory, chunks them,
and for each chunk generates a question-answer pair.
The output is saved as a JSONL file (one JSON object per line).
"""

import time
import json
import argparse
from tqdm import tqdm

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from oci_models import get_llm
from prompts import QA_PROMPT_TEMPLATE
from chunk_utils import generate_chunks_with_metadata
from utils import get_console_logger
from config import QA_MODEL_ID

# configs
FILE_PATTERN = "*.pdf"


logger = get_console_logger()

prompt = PromptTemplate.from_template(QA_PROMPT_TEMPLATE)
# the model for Q/A generation...
llm = get_llm(model_id=QA_MODEL_ID, temperature=0.5)

# LCEL chain: document -> prompt -> LLM -> output
chain = (
    RunnableLambda(lambda d: {"context": d.page_content})
    | prompt
    | llm
    | StrOutputParser()
)


def parse_qa(output: str):
    """Extract question and answer from formatted LLM output."""
    question, answer = None, None
    if "Question:" in output and "Answer:" in output:
        try:
            q_part, a_part = output.split("Answer:", 1)
            question = q_part.replace("Question:", "").strip()
            answer = a_part.strip()
        except Exception:
            pass
    return question, answer


def process_directory(input_dir, output_path):
    """
    Process all the pdf in the provided directory
    """
    # Load and chunk documents
    chunks = generate_chunks_with_metadata(input_dir)

    # Generate Q&A using LLM
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in tqdm(chunks, desc="Generating questions..."):
            # generate the question
            # to avoid to be throttled
            time.sleep(1)

            try:
                # if we fail to process one, go to next chunk
                response = chain.invoke(doc)
                q, a = parse_qa(response)

                if q and a:
                    result = {
                        "question": q,
                        "answer": a,
                        "node_id": doc.metadata["node_id"],
                    }
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.error(
                    "Error processing chunk %s: %s", doc.metadata["node_id"], e
                )
                logger.error("Trying to continue processing...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate golden Q&A set from documents"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to directory containing pdf documents",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="golden_qa.jsonl",
        help="Output JSONL file path",
    )
    args = parser.parse_args()

    # here we do all the processing
    print("")
    print("Generating Q&A dataset using all pdf in directory: ", args.input_dir)
    print("LLM used for Q/A generation is: ", QA_MODEL_ID)
    print("")
    process_directory(args.input_dir, args.output_file)

    print("")
    print(f"Golden Q&A dataset written to {args.output_file}")
