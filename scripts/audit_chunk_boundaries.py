def flag_problem_chunks(docs):
    flagged = []
    for i, doc in enumerate(docs):
        text = doc.page_content.strip()
        # logic to find if the chunk is problematic. For example, if the chunk ends without a period or punctuation or starts with a lowercase letter or number or symbols.
        if not text.endswith('.') and not text.endswith('!') and not text.endswith('?') or text[0].islower() or text[0].isdigit() or not text[0].isalpha():
            flagged.append((i, text))
    return flagged