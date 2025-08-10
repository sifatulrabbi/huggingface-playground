import base64
from PyPDF2 import PdfReader
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_openai import ChatOpenAI
from tiktoken import get_encoding

encoder = get_encoding("o200k_base")


def pdf_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def extract_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if text.strip():
            text += "\n\n"
        text += page.extract_text()
    print(f"Total tokens in the PDF file: {len(encoder.encode(text)):,}")
    return text


llm = ChatOpenAI(base_url="http://127.0.0.1:8089/v1")
chain = (
    RunnablePassthrough.assign(
        pdf_file=lambda _: extract_pdf_text("./tmp/oai_gpt-oss_model_card.pdf")
    )
    | ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                "You are an helpful assistant who helps the user in regular tasks.\n\n**Information you must take into account when helping the user:**\n{pdf_file}\n\n"
            ),
            HumanMessagePromptTemplate.from_template("{input}"),
        ]
    )
    | llm
)

response = chain.invoke(
    {"input": "Give me a brief comparison between gpt-oss-20b and gpt-oss-120b."}
)
print(response)
