import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def get_section_chunks(text, filename):
    section_pattern = r'\n(?=\d+\.\s+[A-Z\s]{3,})'
    sections = re.split(section_pattern, text)
    docs = []
    for i, content in enumerate(sections):
        lines = content.strip().split('\n')
        header = lines[0] if lines else "INTRO/PREAMBLE"
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        chunks = splitter.split_text(content)
        
        for j, chunk in enumerate(chunks):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "section": header,
                    "section_no": i,
                    "chunk_no": j + 1,
                    "total_section_chunks": len(chunks)
                }
            ))
    return docs