import json
import os
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

cores = os.cpu_count() or 2
os.environ['OMP_NUM_THREADS'] = str(max(1, cores // 2))
os.environ['MKL_NUM_THREADS'] = str(max(1, cores // 2))

import torch
from transformers import MarianMTModel, MarianTokenizer

TRANSLATIONS_DIR = Path('translations')
TRANSLATIONS_DIR.mkdir(exist_ok=True)
QUEUE_FILE = Path('translate-queue.txt')
INPUT_DIR = Path('input')
INPUT_DIR.mkdir(exist_ok=True)

MODEL_EN_JA = 'Helsinki-NLP/opus-mt-en-jap'


def load_model(name):
    print(f'loading model: {name}')
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    model.eval()
    return tok, model


def split_text(text, max_len):
    if len(text) <= max_len:
        return [text]
    chunks, rest = [], text
    while len(rest) > max_len:
        cut = max_len
        for sep in ['. ', '。', '! ', '? ']:
            pos = rest.rfind(sep, 0, max_len)
            if pos > max_len * 0.5:
                cut = pos + len(sep)
                break
        chunks.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        chunks.append(rest)
    return chunks


def translate_batch(tok, model, texts, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        chunks_flat = []
        chunk_map = []
        for t in batch:
            cs = split_text(t, 400)
            chunk_map.append(len(cs))
            chunks_flat.extend(cs)
        inputs = tok(chunks_flat, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=512, num_beams=1)
        decoded = [tok.decode(o, skip_special_tokens=True) for o in out]
        pos = 0
        for n in chunk_map:
            results.append(''.join(decoded[pos:pos+n]))
            pos += n
    return results


def ensure_input_text(book_id):
    input_path = INPUT_DIR / f'{book_id}.json'
    if input_path.exists():
        return True
    if not book_id.startswith('gutenberg_'):
        print(f'  unknown source: {book_id}')
        return False
    gid = book_id.replace('gutenberg_', '')
    print(f'  fetching text: {book_id}')
    try:
        r = subprocess.run(
            [sys.executable, 'scripts/fetch_gutenberg.py', gid],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode != 0:
            print(f'  fetch failed: {r.stderr.strip()[:200]}')
            return False
        return input_path.exists()
    except Exception as e:
        print(f'  fetch exception: {e}')
        return False


def translate_book(book_id, tok, model):
    out_path = TRANSLATIONS_DIR / f'{book_id}.json'
    if out_path.exists():
        print(f'  skip (already translated): {book_id}')
        return 'skip'

    if not ensure_input_text(book_id):
        return 'missing'

    input_path = INPUT_DIR / f'{book_id}.json'
    book = json.loads(input_path.read_text(encoding='utf-8'))
    paragraphs = book['paragraphs']
    print(f'translating: {book_id} ({len(paragraphs)} paras)')
    t0 = time.time()
    translated = translate_batch(tok, model, paragraphs)
    out = {
        'bookId': book_id,
        'formatVersion': 3,
        'translator': 'OPUS-MT',
        'modelName': MODEL_EN_JA,
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'translatedTexts': translated
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    print(f'  done ({time.time()-t0:.1f}s)')
    os.system(f'git add {out_path} && git commit -m "translate: {book_id}" --quiet && git push --quiet')
    return 'done'


def main():
    max_books = 50
    if '--max-books' in sys.argv:
        max_books = int(sys.argv[sys.argv.index('--max-books')+1])
    deadline = time.time() + 5.5 * 3600
    queue = []
    if QUEUE_FILE.exists():
        queue = [l.strip() for l in QUEUE_FILE.read_text().split('\n') if l.strip()]
    done_ids = {p.stem for p in TRANSLATIONS_DIR.glob('*.json')}
    todo = [q for q in queue if q not in done_ids][:max_books]
    if not todo:
        print('nothing to translate')
        return
    print(f'targets: {len(todo)}')
    tok, model = load_model(MODEL_EN_JA)
    count = 0
    for book_id in todo:
        if time.time() > deadline:
            print('timeout')
            break
        try:
            r = translate_book(book_id, tok, model)
            if r == 'done':
                count += 1
        except Exception as e:
            print(f'  fail: {book_id}: {e}')
    print(f'finished: {count}')


if __name__ == '__main__':
    main()
