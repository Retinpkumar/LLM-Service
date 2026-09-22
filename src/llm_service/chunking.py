from langchain_text_splitters import RecursiveCharacterTextSplitter

# Everythin in a single function named chunk_document(text, headers)
def chunk_document(text, headers, source_file, overlap=0):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=overlap,
        add_start_index=True,
    )

    docs = splitter.create_documents([text])

    positions = []
    for header in headers:
        pos = text.find(f"{header}\n")
        positions.append(pos)

    header_positions = list(zip(positions, headers))

    def get_section_for_chunk(start_index, header_positions):
        current_section = None
        for pos, header in header_positions:
            if pos <= start_index:
                current_section = header
            else:
                break
        return current_section

    for i, doc in enumerate(docs):
        section = get_section_for_chunk(doc.metadata["start_index"], header_positions)
        doc.metadata["section"] = section
        doc.metadata["source"] =  source_file
        doc.metadata["chunk_index"] = i

    return docs