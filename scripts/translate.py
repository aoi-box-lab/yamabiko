#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time
import gc
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

# --- スレッドを1に固定 ---
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

from transformers import MarianMTModel, MarianTokenizer

TRANSLATIONS_DIR = Path('translations')
TRANSLATIONS_DIR.mkdir(exist_ok=True)
FAILED_FILE = Path('translate-failed.txt')
INPUT_DIR = Path('input')
INPUT_DIR.mkdir(exist_ok=True)

MODEL_EN_JA = 'Helsinki-NLP/opus-mt-en-jap'
MAX_FAILED = 3
RETRIES = 2
RETRY_WAIT = 5

USER_AGENT = 'Yamabiko/1.0 (https://github.com/aoi-box-lab/yamabiko)'


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ---------------- gutendex から本を探す ----------------

def fetch_popular_ids(limit=2000):
    """gutendex から英語の人気本IDを取ってくる"""
    ids = []
    url = 'https://gutendex.com/books?sort=popular'
    page = 0
    while url and len(ids) < limit:
        page += 1
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': USER_AGENT,
                'Accept': 'application/json',
            })
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
        except Exception as e:
            log(f'  gutendex fetch failed (page {page}): {e}')
            break
        for b in data.get('results', []):
            if 'en' not in b.get('languages', []):
                continue
            formats = b.get('formats', {})
            if not any(k.startswith('text/plain') or k.startswith('text/html')
                       for k in formats):
                continue
            ids.append(f"gutenberg_{b['id']}")
        log(f'  page {page}: total {len(ids)}')
        url = data.get('next')
    return ids


# ---------------- テキスト分割・翻訳 ----------------

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


def translate_batch(tok, model, texts):
    results = []
    for i, text in enumerate(texts):
        chunks = split_text(text, 400)
        decoded = []
        for chunk in chunks:
            inputs = tok(chunk, return_tensors='pt',
                         truncation=True, max_length=512)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    num_beams=1,
                    do_sample=False,
                )
            decoded.append(tok.decode(out[0], skip_special_tokens=True))
            del inputs, out
        results.append(''.join(decoded))
        del decoded
        if (i + 1) % 50 == 0:
            log(f'    progress: {i+1}/{len(texts)}')
            gc.collect()
    gc.collect()
    return results


# ---------------- 取得 ----------------

def ensure_input_text(book_id):
    input_path = INPUT_DIR / f'{book_id}.json'
    if input_path.exists():
        return True
    if not book_id.startswith('gutenberg_'):
        log(f'  unknown source: {book_id}')
        return False
    gid = book_id.replace('gutenberg_', '')
    log(f'  fetching text: {book_id}')
    try:
        r = subprocess.run(
            [sys.executable, 'scripts/fetch_gutenberg.py', gid],
            capture_output=True, text=True, timeout=120
        )
        if r.returncode != 0:
            log(f'  fetch failed: {r.stderr.strip()[:200]}')
            return False
        return input_path.exists()
    except Exception as e:
        log(f'  fetch exception: {e}')
        return False


# ---------------- 1冊翻訳 ----------------

def try_translate(book_id, tok, model):
    out_path = TRANSLATIONS_DIR / f'{book_id}.json'
    if out_path.exists():
        log(f'  skip (already translated): {book_id}')
        return 'skip'

    for attempt in range(RETRIES + 1):
        try:
            if not ensure_input_text(book_id):
                return 'fail'

            book = json.loads((INPUT_DIR / f'{book_id}.json').read_text(encoding='utf-8'))
            paragraphs = book['paragraphs']
            log(f'translating: {book_id} ({len(paragraphs)} paras)')
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
            log(f'  done ({time.time()-t0:.1f}s): {book_id}')
            del translated, book, paragraphs
            gc.collect()
            return 'done'
        except Exception as e:
            log(f'  attempt {attempt+1}/{RETRIES+1} failed: {book_id}: {e}')
            gc.collect()
            if attempt < RETRIES:
                time.sleep(RETRY_WAIT)
    return 'fail'


# ---------------- 失敗リスト ----------------

def load_failed():
    result = {}
    if FAILED_FILE.exists():
        for line in FAILED_FILE.read_text(encoding='utf-8').split('\n'):
            parts = line.strip().split()
            if len(parts) == 2 and parts[1].isdigit():
                result[parts[0]] = int(parts[1])
    return result


def save_failed(failed):
    lines = [f'{k} {v}' for k, v in failed.items() if v < MAX_FAILED]
    FAILED_FILE.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')


def load_model(name):
    log(f'loading model: {name}')
    tok = MarianTokenizer.from_pretrained(name)
    model = MarianMTModel.from_pretrained(name)
    model.eval()
    return tok, model


# ---------------- main ----------------

def main():
    max_books = 2
    if '--max-books' in sys.argv:
        max_books = int(sys.argv[sys.argv.index('--max-books') + 1])

    max_minutes = 25
    if '--max-minutes' in sys.argv:
        max_minutes = int(sys.argv[sys.argv.index('--max-minutes') + 1])
    deadline = time.time() + max_minutes * 60

    done = {p.stem for p in TRANSLATIONS_DIR.glob('*.json') if p.name != 'manifest.json'}
    failed = load_failed()

    log('fetching popular list from gutendex...')
    popular = fetch_popular_ids(2000)
    log(f'  fetched: {len(popular)}')

    # 優先: 失敗リスト（まだ done でない） → 人気順
    priority = [b for b in failed.keys() if b not in done]
    normal = [b for b in popular if b not in done and b not in priority]
    todo = (priority + normal)[:max_books]

    if not todo:
        log('nothing to translate')
        return

    log(f'targets: {todo}')
    tok, model = load_model(MODEL_EN_JA)

    done_now, failed_now = [], []
    for book_id in todo:
        if time.time() > deadline:
            log('timeout reached, stopping')
            break
        result = try_translate(book_id, tok, model)
        if result in ('done', 'skip'):
            done_now.append(book_id)
        else:
            failed_now.append(book_id)

    new_failed = {k: v for k, v in failed.items() if k not in done_now}
    for b in failed_now:
        new_failed[b] = new_failed.get(b, 0) + 1
    save_failed(new_failed)

    log(f'finished: {len(done_now)} done, {len(failed_now)} failed')

    if done_now:
        try:
            subprocess.run(['git', 'add', '-A'], check=False)
            subprocess.run(
                ['git', 'commit', '-m', f'translate: {", ".join(done_now)}', '--quiet'],
                check=False
            )
            subprocess.run(['git', 'push', '--quiet'], check=False)
            log('pushed')
        except Exception as e:
            log(f'git error: {e}')


if __name__ == '__main__':
    main()
