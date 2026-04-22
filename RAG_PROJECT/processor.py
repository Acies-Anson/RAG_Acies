import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Regex pattern:
# Split when a new line is followed by something like:
# 1. INTRODUCTION
# 2. TERMS AND CONDITIONS
SECTION_PATTERN = r'\n(?=\d+\.\s+[A-Z\s]{3,})'

# Reusable text splitter configuration
TEXT_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)


def get_section_chunks(text: str, filename: str) -> list[Document]:
    """
    Split a document into logical sections first, then split each section
    into smaller overlapping chunks for retrieval.

    Args:
        text: Full text content of the file.
        filename: Source file name.

    Returns:
        List of LangChain Document objects with metadata.
    """

    # Split document into sections using numbered uppercase headings
    sections = re.split(SECTION_PATTERN, text)

    documents = []

    for section_index, section_content in enumerate(sections):
        cleaned_section = section_content.strip()
        if not cleaned_section:
            continue

        # First line is treated as section header
        lines = cleaned_section.split("\n")
        header = lines[0] if lines else "INTRO/PREAMBLE"

        # Split section into smaller chunks
        chunks = TEXT_SPLITTER.split_text(cleaned_section)
        total_chunks = len(chunks)

        for chunk_index, chunk in enumerate(chunks, start=1):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": filename,
                        "section": header,
                        "section_no": section_index,
                        "chunk_no": chunk_index,
                        "total_section_chunks": total_chunks,
                    },
                )
            )

    return documents
