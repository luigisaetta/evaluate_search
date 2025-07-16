"""
Prompts

You can work here, for example, to improve the quality of the synthetic dataset generated.
"""

QA_PROMPT_TEMPLATE = """
You are a helpful AI teacher.
Given the following document excerpt, generate a meaningful and generic question that can be answered 
using the information contained in the excerpt.
Do not include details specific to the format or location of the excerpt (e.g., page number).
Do not refer explicitly to the excerpt or mention that it comes from a document.
Avoid questions that simply repeat named entities (e.g., names, places, products) without requiring contextual understanding.
Craft the question so that it would make sense to someone unfamiliar with the document.
Ensure the question is answerable based on the excerpt.
Make the question in the same language as the excerpt.

For example, avoid questions like:
* Qual è lo scopo delle tecnologie menzionate nel testo? (which text we're referring to?)
* Quali sono i vantaggi offerti a chi sceglie i prodotti e i servizi descritti nel testo? (again, which text?)
* Quali dispositivi sono menzionati in relazione al tema trattato? (too generic, what is the theme mentioned? we don't know.)

Excerpt:
\"\"\"{context}\"\"\"

Format:
Question: <your question>
Answer: <the answer>
"""
