from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from jinja2 import Template
from langchain.embeddings import OpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import Document
from langchain.vectorstores import FAISS

from .config import Settings
from .models import LabelResponse
from .openrouter import OpenRouterClient


def build_vector_store(df: pd.DataFrame, text_column: str, settings: Settings) -> FAISS:
    """Create a FAISS vector store from dataframe text column."""
    embeddings = OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model=settings.embedding_model,
    )
    docs = [
        Document(page_content=str(row[text_column]), metadata=row.to_dict())
        for _, row in df.iterrows()
    ]
    return FAISS.from_documents(docs, embeddings)


def get_similar_examples(store: FAISS, text: str, k: int = 3) -> list[Document]:
    """Retrieve similar examples from the vector store."""
    if store is None:
        return []
    return store.similarity_search(text, k=k)


class AutoLabeler:
    """High-level autolabeling pipeline."""

    def __init__(self, settings: Settings, template_path: Path | None = None) -> None:
        self.settings = settings
        self.llm = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            model=settings.llm_model,
        )
        self.parser = PydanticOutputParser(pydantic_object=LabelResponse)
        if template_path is None:
            template_path = Path(__file__).parent / "templates" / "label_prompt.jinja"
        self.template = Template(template_path.read_text())

    def label_text(
        self, text: str, examples: Iterable[Document] | None = None
    ) -> LabelResponse:
        rendered = self.template.render(
            text=text,
            examples=list(examples or []),
            format_instructions=self.parser.get_format_instructions(),
        )
        response = self.llm.invoke(rendered)
        return self.parser.parse(response.content)

    def label_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str,
        label_column: str = "label",
        use_embeddings: bool = False,
        k: int = 3,
    ) -> pd.DataFrame:
        store = None
        if use_embeddings:
            store = build_vector_store(df, text_column, self.settings)
        results = []
        for _, row in df.iterrows():
            query = str(row[text_column])
            examples = get_similar_examples(store, query, k=k) if store else []
            result = self.label_text(query, examples)
            result_dict = row.to_dict()
            result_dict[label_column] = result.label
            if result.metadata:
                result_dict.update({f"meta_{k}": v for k, v in result.metadata.items()})
            results.append(result_dict)
        return pd.DataFrame(results)

    def label_csv(
        self,
        input_file: Path,
        output_file: Path,
        text_column: str,
        label_column: str = "label",
        use_embeddings: bool = False,
    ) -> None:
        df = pd.read_csv(input_file)
        labeled = self.label_dataframe(df, text_column, label_column, use_embeddings)
        labeled.to_csv(output_file, index=False, quoting=csv.QUOTE_NONNUMERIC)
