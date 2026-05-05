import io
import re

import pdfplumber
import pymorphy3
import replicate
from PIL import Image
import nltk
from nltk.corpus import words as nltk_words

# =============================================================================
# Извлечение текста документов в формате .pdf c учётом таблиц
# =============================================================================

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

_ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())

def call_dots_ocr(image: Image.Image) -> str:
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    output = replicate.run(
        "rednote-ai/dots-ocr:dots-ocr-1.5",
        input={
            "image": img_bytes,
            "prompt_mode": "prompt_layout_all_en"
        }
    )
    if isinstance(output, str):
        return output
    else:
        return ' '.join(output)

def is_english_word(word: str) -> bool:
    return word.lower() in _ENGLISH_WORDS

def is_potential_abbr_part(token, morph_ru):
    if not token:
        return False
    if not re.fullmatch(r'[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)*', token):
        return False
    if morph_ru.word_is_known(token) or is_english_word(token):
        return False
    return any(c.isupper() or c.isdigit() for c in token)

def has_camel_case(token):
    for j in range(len(token)-1):
        if token[j].islower() and token[j+1].isupper():
            return True
    return False

def is_single_letter(token):
    return len(token) == 1 and token.isalpha()

def merge_split_words(text: str) -> str:
    morph_ru = pymorphy3.MorphAnalyzer()
    words = text.split()
    RU_SERVICE_WORDS = {
        'с', 'в', 'к', 'у', 'о', 'и', 'а', 'не', 'ни',
        'на', 'по', 'за', 'до', 'из', 'от', 'без', 'под',
        'об', 'обо', 'ко', 'во', 'со', 'ото'
    }
    if len(words) < 2:
        return text
    merged = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            candidate = words[i] + words[i + 1]
            if morph_ru.word_is_known(candidate):
                if not (words[i].lower() in RU_SERVICE_WORDS or words[i + 1].lower() in RU_SERVICE_WORDS):
                    merged.append(candidate)
                    print(f"DEBUG: merging '{words[i]}' + '{words[i + 1]}' -> '{candidate}'", flush=True)
                    i += 2
                    continue
            if is_english_word(candidate):
                print(f"DEBUG: merging '{words[i]}' + '{words[i + 1]}' -> '{candidate}'", flush=True)
                merged.append(candidate)
                i += 2
                continue
            if words[i].endswith('-') and not words[i+1].startswith('-'):
                if not (is_english_word(candidate) or morph_ru.word_is_known(candidate)):
                    if is_potential_abbr_part(words[i].rstrip('-')) or is_potential_abbr_part(words[i + 1]):
                        merged.append(candidate)
                        print(f"DEBUG: merging ABBR-DASH '{words[i]}' + '{words[i + 1]}' -> '{candidate}'", flush=True)
                        i += 2
                        continue
            if (is_potential_abbr_part(words[i]) and is_potential_abbr_part(words[i + 1])) or \
                    (is_potential_abbr_part(words[i]) and is_single_letter(words[i + 1])) or \
                    (is_single_letter(words[i]) and is_potential_abbr_part(words[i + 1])):
                if not (is_english_word(candidate) or morph_ru.word_is_known(candidate)):
                    if (has_camel_case(words[i]) or has_camel_case(words[i + 1]) or
                            is_single_letter(words[i]) or is_single_letter(words[i + 1]) or
                            (words[i][0].isupper() and words[i + 1][0].isupper())):
                        merged.append(candidate)
                        print(f"DEBUG: merging ABBR '{words[i]}' + '{words[i + 1]}' -> '{candidate}'", flush=True)
                        i += 2
                        continue
        merged.append(words[i])
        i += 1
    return ' '.join(merged)

def extract_text(path):
    pages_text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            lines = page.extract_text_lines()
            tables = page.find_tables()
            table_bboxes = [t.bbox for t in tables]

            if not tables:
                text = page.extract_text()
                if text:
                    pages_text.append(text)
                continue
            all_items = []
            for line in lines:
                inside = False
                for tx0, ty0, tx1, ty1 in table_bboxes:
                    if (line['x0'] >= tx0 and line['x1'] <= tx1 and
                        line['top'] >= ty0 and line['bottom'] <= ty1):
                        inside = True
                        break
                if not inside:
                    all_items.append((line['top'], line['text']))
            for t in tables:
                cells = t.extract()
                if not cells:
                    continue
                cell_bboxes = t.cells if hasattr(t, 'cells') else None
                rows_text = []
                for i, row in enumerate(cells):
                    row_parts = []
                    for j, cell in enumerate(row):
                        if cell is None:
                            row_parts.append('')
                            continue
                        cell_clean = ' '.join(cell.split())
                        if ' ' not in cell_clean and len(cell_clean) > 15:
                            if cell_bboxes and i < len(cell_bboxes) and j < len(cell_bboxes[i]):
                                bbox_raw = cell_bboxes[i][j]
                                if isinstance(bbox_raw, dict):
                                    bbox = (bbox_raw.get('x0'), bbox_raw.get('top'),
                                            bbox_raw.get('x1'), bbox_raw.get('bottom'))
                                elif isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
                                    bbox = tuple(bbox_raw)
                                else:
                                    bbox = None
                                if bbox and all(isinstance(v, (int, float)) for v in bbox):
                                    try:
                                        cell_img = page.within_bbox(bbox).to_image(resolution=150).original
                                        ocr_text = call_dots_ocr(cell_img)
                                        if ocr_text:
                                            cell_clean = ' '.join(ocr_text.split())
                                    except Exception as e:
                                        print(f"OCR failed for cell ({i},{j}): {e}")
                        row_parts.append(cell_clean)
                    row_text = ' '.join(row_parts).strip()
                    if row_text:
                        row_text = merge_split_words(row_text)
                        rows_text.append(row_text)
                y0 = t.bbox[1]
                for idx, row_text in enumerate(rows_text):
                    all_items.append((y0 + idx * 0.1, row_text))
            all_items.sort(key=lambda x: x[0])
            page_text = '\n'.join(text for _, text in all_items)
            pages_text.append(page_text)
    text = '\n'.join(pages_text)
    text = re.sub(r"([А-Яа-яЁёA-Za-z])\-\s*\n\s*([А-Яа-яЁёA-Za-z])", r"\1\2", text)
    return text