import re


def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\bedit\b', '', text, flags=re.IGNORECASE)

    nav_patterns = [
        r'Jump to content',
        r'From Wikipedia',
        r'From Wikipedia the free encyclopedia',
        r'- wikipedia the free encyclopedia',
        r'wikipedia the free encyclopedia',
        r'the free encyclopedia',
        r'- Wikipedia',
        r'For the journal see',
        r'File history',
        r'File usage',
        r'Global file usage',
        r'Metadata',
        r'Size of this preview',
        r'Other resolutions',
        r'Original file',
        r'MIME type',
        r'This is a file',
        r'Main article:',
        r'Main articles:',
        r'See also:',
        r'Further information:',
        r'\bv t e\b',
        r'Part of a series on',
    ]
    for pattern in nav_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    text = text.lower()
    text = re.sub(r'[^\w\s\.\%\-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text
