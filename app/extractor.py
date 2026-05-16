import io
import re

import pdfplumber
import pymorphy3
import replicate
from PIL import Image
import nltk
from nltk.corpus import words as nltk_words
from app.settings import Settings

# =============================================================================
# Извлечение текста документов в формате .pdf c учётом таблиц
# =============================================================================

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

_ENGLISH_WORDS = set(w.lower() for w in nltk_words.words())
_RUSSIAN_WORDS = {'пао', 'ртк', 'ооо', 'мрф'}
RU_SERVICE_WORDS = {
    'с', 'в', 'к', 'у', 'о', 'и', 'а', 'не', 'ни',
    'на', 'по', 'за', 'до', 'из', 'от', 'без', 'под',
    'об', 'обо', 'ко', 'во', 'со', 'ото', 'же', 'бы', 'ли',
    'то', 'что', 'как', 'мы', 'вы', 'он', 'она', 'они', 'его', 'ее', 'их',
    'но', 'да', 'или', 'уже', 'еще', 'это', 'все', 'всё', 'так', 'для',
    'от', 'при', 'про', 'над', 'перед', 'через', 'между', 'около', 'вокруг',
    'из-за', 'из-под', 'и т.д.', 'и т.п.'
}
settings = Settings()
morph_ru = pymorphy3.MorphAnalyzer()

def _has_mixed_scripts_no_camel(text):
    has_cyrillic = bool(re.search(r'[а-яё]', text, re.I))
    has_latin = bool(re.search(r'[a-z]', text, re.I))
    if not (has_cyrillic and has_latin):
        return False
    if re.search(r'[a-z][A-Z]|[а-яё][A-Z]|[a-z][А-ЯЁ]', text):
        return False  # есть граница, ок
    return True
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

import re

def fix_compound_breaks(text: str) -> str:
    text = re.sub(
        r'(\w+-)\s+(\w+)\s+(\w+)',
        lambda m: f'{m.group(1)}{m.group(2)}{m.group(3).lower()}'
        if m.group(2).isalpha() and len(m.group(2)) <= 3 and m.group(3).isalpha()
        else m.group(0),
        text
    )
    text = re.sub(
        r'\b([A-Z][a-z]+)\s+([A-Z][a-z]?)\s+([a-z]{1,2})\b',
        lambda m: f'{m.group(1)}{m.group(2)}{m.group(3)}'
        if m.group(2)[0].isupper() and len(m.group(3)) <= 2
        else m.group(0),
        text
    )
    text = re.sub(
        r'\b([A-Z][a-z]+)\s+([A-Z][a-z]{1,2})\s+([a-z]{2,})\b',
        lambda m: f'{m.group(1)}{m.group(2)}{m.group(3)}'
        if all(c.isalpha() for c in m.group(2) + m.group(3))
        else m.group(0),
        text
    )
    text = re.sub(
        r'\b([A-Z][a-z]+)\s+([a-z])([A-Z][a-z]+)\b',
        lambda m: f'{m.group(1)}{m.group(2).upper()}{m.group(3)}'
        if len(m.group(2)) == 1 and m.group(2).islower()
        else m.group(0),
        text
    )
    return text

def is_abbr(token):
    return bool(re.fullmatch(r'[A-ZА-ЯЁ0-9](?:[.A-ZА-ЯЁ0-9])*', token) and
                any(c.isupper() for c in token))

def is_camel_or_pascal(token):
    if not token:
        return False
    if not token[0].isupper():
        return False
    has_lower = any(c.islower() for c in token)
    if has_lower:
        for j in range(len(token) - 1):
            if token[j].islower() and token[j + 1].isupper():
                return True  # camelCase
        return True
    return False

def is_single_letter(token, RU_SERVICE_WORDS):
    return len(token) == 1 and token.isalpha() and ((token.lower() in RU_SERVICE_WORDS)==False)

def merge_split_words(text: str) -> str:
    words = [w for w in text.split() if w]
    if len(words) < 2:
        return text

    i = 0
    while i < len(words) - 1:
        w1 = words[i]
        w2 = words[i + 1]
        candidate = w1 + w2

        if morph_ru.word_is_known(candidate):
            if w1.lower() not in RU_SERVICE_WORDS and w2.lower() not in RU_SERVICE_WORDS:
                words[i] = candidate
                words.pop(i + 1)
                continue
            elif (w1.lower() in RU_SERVICE_WORDS and not morph_ru.word_is_known(w2)) or \
                 (w2.lower() in RU_SERVICE_WORDS and not morph_ru.word_is_known(w1)):
                words[i] = candidate
                words.pop(i + 1)
                continue

        w2_alpha = w2.rstrip('?.,;:!)»')
        candidate_clean = w1 + w2_alpha
        if is_english_word(candidate_clean):
            words[i] = candidate_clean + w2[len(w2_alpha):]
            words.pop(i + 1)
            continue

        if w1.endswith('-') and not w2.startswith('-') and len(w1) > 1:
            candidate_no_dash = w1.rstrip('-') + w2
            if morph_ru.word_is_known(candidate_no_dash) or is_english_word(candidate_no_dash):
                words[i] = candidate_no_dash
                words.pop(i + 1)
                continue

        can_merge = False
        if _has_mixed_scripts_no_camel(candidate):
            can_merge = False
        if is_abbr(w1) and is_abbr(w2):
            can_merge = True
        elif is_single_letter(w1, RU_SERVICE_WORDS) and (is_abbr(w2) or is_camel_or_pascal(w2)):
            can_merge = True
        elif is_single_letter(w2, RU_SERVICE_WORDS) and (is_abbr(w1) or is_camel_or_pascal(w1)):
            can_merge = True
        elif (is_camel_or_pascal(w1) and (is_abbr(w2) or is_camel_or_pascal(w2))) or \
             (is_camel_or_pascal(w2) and (is_abbr(w1) or is_camel_or_pascal(w1))):
            can_merge = True
        elif w1[0].isupper() and w2[0].isupper() and not any(c.islower() for c in w1) and not any(c.islower() for c in w2):
            can_merge = True
        if w1.lower() in _RUSSIAN_WORDS or w2.lower() in _RUSSIAN_WORDS:
            can_merge = False
        if is_english_word(candidate) and bool(re.search(r'[а-яё]', candidate, re.I)):
            can_merge = False

        if can_merge:
            if not (is_english_word(candidate) or morph_ru.word_is_known(candidate)):
                if not (any(c.islower() for c in w1) and w2[0].isupper() and len(w1) > 1 and len(w2) > 1):
                    words[i] = candidate
                    words.pop(i + 1)
                    continue
        i += 1
    return fix_compound_breaks(' '.join(words))

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
                num_cols = len(cells[0]) if cells else 0
                rows_text = []
                for i, row in enumerate(cells):
                    row_parts = []
                    for j, cell in enumerate(row):
                        if cell is None:
                            row_parts.append('')
                            continue
                        cell_clean = ' '.join(cell.split())
                        has_long_unknown = any(
                            len(tok) > 15
                            and re.fullmatch(r'[а-яёА-ЯЁa-zA-Z]+', tok)
                            and not morph_ru.word_is_known(tok)
                            and not is_english_word(tok)
                            for tok in cell_clean.split()
                        )
                        if has_long_unknown:
                            cell_idx = i * num_cols + j
                            if t.cells and cell_idx < len(t.cells):
                                bbox_raw = t.cells[cell_idx]
                                if isinstance(bbox_raw, (list, tuple)) and len(bbox_raw) == 4:
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
                                        print(f"Ошибка OCR в ячейке ({i},{j}): {e}")
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
    text = re.sub(r'\?([А-ЯЁA-Za-z])', r'? \1', text)
    return text