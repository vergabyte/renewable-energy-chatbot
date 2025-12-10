import re


def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\bedit\b', '', text, flags=re.IGNORECASE)

    nav_patterns = [
        'Jump to content',
        'From Wikipedia',
        'From Wikipedia the free encyclopedia',
        'Main article:',
        'Main articles:',
        'See also:',
        'Further information:',
        r'\bv t e\b',
        'Part of a series on',
    ]
    for pattern in nav_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = text.lower()
    text = re.sub(r'[^\w\s\.\%\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
