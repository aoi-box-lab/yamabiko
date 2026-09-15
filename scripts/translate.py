#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time
import gc
import urllib.request
from pathlib import Path
from datetime import datetime, timezone

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import ctranslate2
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

TRANSLATIONS_DIR = Path('translations')
TRANSLATIONS_DIR.mkdir(exist_ok=True)
QUEUE_FILE = Path('translate-queue.txt')
FAILED_FILE = Path('translate-failed.txt')
INPUT_DIR = Path('input')
INPUT_DIR.mkdir(exist_ok=True)

CT2_MODEL_ID = 'olob0/nllb-200-distilled-600M-ct2-int8_float16'
SRC_LANG = 'eng_Latn'
TGT_LANG = 'jpn_Jpan'

MAX_FAILED = 3
RETRIES = 2
RETRY_WAIT = 5

USER_AGENT = 'Yamabiko/1.0 (https://github.com/aoi-box-lab/yamabiko)'
GUTENBERG_TOP = 'https://www.gutenberg.org/browse/scores/top'

GUTENBERG_MIRRORS = [
    'https://www.gutenberg.org',
    'https://gutenberg.pglaf.org',
    'https://mirrors.xmission.com/gutenberg',
]

# 本文URLの候補パターン（複数形式に対応）
TEXT_URL_PATTERNS = [
    '/cache/epub/{id}/pg{id}.txt',
    '/files/{id}/{id}-0.txt',
    '/files/{id}/{id}.txt',
    '/ebooks/{id}.txt.utf-8',
]


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ---------------- Gutenberg 直アクセス ----------------

def fetch_top_html():
    req = urllib.request.Request(GUTENBERG_TOP, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode('utf-8', errors='replace')


def extract_book_ids(html, limit=200):
    ids = []
    for m in re.finditer(r'/ebooks/(\d+)', html):
        gid = m.group(1)
        book_id = f'gutenberg_{gid}'
        if book_id not in ids:
            ids.append(book_id)
        if len(ids) >= limit:
            break
    return ids


def fetch_popular_ids(limit=200):
    try:
        html = fetch_top_html()
    except Exception as e:
        log(f'  gutenberg top page failed: {e}')
        return []
    return extract_book_ids(html, limit)


def fetch_text(book_id):
    gid = book_id.replace('gutenberg_', '')
    for base in GUTENBERG_MIRRORS:
        for pat in TEXT_URL_PATTERNS:
            url = base + pat.format(id=gid)
            try:
                req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read().decode('utf-8', errors='replace')
                log(f'    fetched: {url}')
                return data
            except Exception as e:
                log(f'    {url} failed: {e}')
                continue
    return None


# ---------------- debug ----------------

def debug_top_book():
    log('fetching gutenberg top page...')
    try:
        html = fetch_top_html()
    except Exception as e:
        log(f'  top page failed: {e}')
        return

    ids = extract_book_ids(html, limit=5)
    log(f'  top ids: {ids}')

    if not ids:
        log('no ids found')
        return

    book_id = ids[0]
    gid = book_id.replace('gutenberg_', '')
    log(f'=== top book: {book_id} ===')
    log(f'  sample url: https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt')

    text = fetch_text(book_id)
    if not text:
        log('text fetch failed')
        return

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    log()
    log('=== raw text (first 2000 chars) ===')
    log(text[:2000])
    log()
    log(f'=== raw length: {len(text)} chars ===')


# ---------------- queue ----------------

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


def build_queue(max_books):
    done = {p.stem for p in TRANSLATIONS_DIR.glob('*.json') if p.name != 'manifest.json'}
    failed = load_failed()

    log('fetching popular list from gutenberg top page...')
    popular = fetch_popular_ids(200)
    log(f'  fetched: {len(popular)}')

    priority = [b for b in failed.keys() if b not in done]
    normal = [b for b in popular if b not in done and b not in priority]
    todo = (priority + normal)[:max_books]

    QUEUE_FILE.write_text('\n'.join(todo) + ('\n' if todo else ''), encoding='utf-8')
    log(f'queue: {todo}')
    return todo


# ---------------- 翻訳 ----------------

def split_text(text, max_len=400):
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


def translate_batch(tok, translator, texts):
    results = []
    for i, text in enumerate(texts):
        chunks = split_text(text)
        source_tokens = [
            tok.convert_ids_to_tokens(tok.encode(chunk, truncation=True))
            for chunk in chunks
        ]
        target_prefix = [[TGT_LANG]] * len(source_tokens)

        out = translator.translate_batch(
            source_tokens,
            target_prefix=target_prefix,
            beam_size=1,
            max_decoding_length=256,
            repetition_penalty=1.2,
        )

        decoded = []
        for r in out:
            tokens = r.hypotheses[0]
            if tokens and tokens[0] == TGT_LANG:
                tokens = tokens[1:]
            decoded.append(tok.decode(tok.convert_tokens_to_ids(tokens)))

        results.append(''.join(decoded))
        del decoded, source_tokens, out
        if (i + 1) % 20 == 0:
            log(f'    progress: {i+1}/{len(texts)}')
            gc.collect()
    gc.collect()
    return results


def ensure_input_text(book_id):
    input_path = INPUT_DIR / f'{book_id}.json'
    if input_path.exists():
        return True

    log(f'  fetching text: {book_id}')
    text = fetch_text(book_id)
    if not text:
        log('  fetch failed')
        return False

    text = text.replace('\r\n', '\n').replace('\r', '\n')
    si = text.find('*** START OF')
    ei = text.find('*** END OF')
    if si > 0:
        text = text[text.find('\n', si) + 1:]
    if ei > 0:
        text = text[:ei]

    paragraphs = []
    for p in re.split(r'\n\s*\n', text):
        p = p.strip()
        if len(p) < 2:
            continue
        p = re.sub(r'\s+', ' ', p).strip()
        if len(p) > 1:
            paragraphs.append(p)

    if not paragraphs:
        log('  no paragraphs')
        return False

    book = {
        'id': book_id,
        'title': '',
        'author': '',
        'sourceLang': 'en',
        'paragraphs': paragraphs,
    }
    input_path.write_text(json.dumps(book, ensure_ascii=False), encoding='utf-8')
    log(f'  saved: {len(paragraphs)} paras')
    return True


def try_translate(book_id, tok, translator):
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
            translated = translate_batch(tok, translator, paragraphs)

            out = {
                'bookId': book_id,
                'formatVersion': 3,
                'translator': 'NLLB-200 (CTranslate2)',
                'modelName': CT2_MODEL_ID,
                'generatedAt': datetime.now(timezone.utc).isoformat(),
                'translatedTexts': translated,
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


# ---------------- git ----------------

def git_push_with_retry(max_retries=5):
    for i in range(max_retries):
        r = subprocess.run(['git', 'push', '--quiet'], capture_output=True)
        if r.returncode == 0:
            return True
        log(f'  push retry {i+1}/{max_retries}')
        subprocess.run(['git', 'pull', '--rebase', '--quiet'], check=False)
        time.sleep(5)
    return False


# ---------------- main ----------------

def load_model():
    log(f'loading CTranslate2 model: {CT2_MODEL_ID}')
    model_path = snapshot_download(CT2_MODEL_ID)
    log(f'  model path: {model_path}')

    tok = AutoTokenizer.from_pretrained(model_path)
    translator = ctranslate2.Translator(
        model_path,
        device='cpu',
        compute_type='int8',
        inter_threads=1,
        intra_threads=1,
    )
    return tok, translator


def main():
    if '--debug-top' in sys.argv:
        debug_top_book()
        return

    max_books = 2
    if '--max-books' in sys.argv:
        max_books = int(sys.argv[sys.argv.index('--max-books') + 1])

    max_minutes = 25
    if '--max-minutes' in sys.argv:
        max_minutes = int(sys.argv[sys.argv.index('--max-minutes') + 1])
    deadline = time.time() + max_minutes * 60

    shard = 0
    shards = 1
    if '--shard' in sys.argv:
        shard = int(sys.argv[sys.argv.index('--shard') + 1])
    if '--shards' in sys.argv:
        shards = int(sys.argv[sys.argv.index('--shards') + 1])

    todo_all = build_queue(max_books * shards)
    if not todo_all:
        log('nothing to translate')
        return

    todo = [b for i, b in enumerate(todo_all) if i % shards == shard]
    if not todo:
        log(f'no work for shard {shard}/{shards}')
        return

    log(f'targets: {todo}')
    tok, translator = load_model()

    done_now, failed_now = [], []
    for book_id in todo:
        if time.time() > deadline:
            log('timeout reached, stopping')
            break
        result = try_translate(book_id, tok, translator)
        if result in ('done', 'skip'):
            done_now.append(book_id)
        else:
            failed_now.append(book_id)

    failed = load_failed()
    new_failed = {k: v for k, v in failed.items() if k not in done_now}
    for b in failed_now:
        new_failed[b] = new_failed.get(b, 0) + 1
    save_failed(new_failed)

    log(f'finished: {len(done_now)} done, {len(failed_now)} failed')

    if done_now:
        subprocess.run(['git', 'add', '-A'], check=False)
        subprocess.run(
            ['git', 'commit', '-m', f'translate: {", ".join(done_now)}', '--quiet'],
            check=False,
        )
        ok = git_push_with_retry()
        log(f'push: {"ok" if ok else "failed"}')


if __name__ == '__main__':
    main()
