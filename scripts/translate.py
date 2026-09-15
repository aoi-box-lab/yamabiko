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


def log(*args, **kwargs):
    print(*args, **kwargs, flush=True)


# ---------------- gutendex ----------------

def fetch_popular_ids(limit=200):
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
            with urllib.request.urlopen(req, timeout=30) as r:
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

    log('fetching popular list from gutendex...')
    popular = fetch_popular_ids(200)
    log(f'  fetched: {len(popular)}')

    priority = [b for b in failed.keys() if b not in done]
    normal = [b for b in popular if b not in done and b not in priority]
    todo = (priority + normal)[:max_books]

    QUEUE_FILE.write_text('\n'.join(todo) + ('\n' if todo else ''), encoding='utf-8')
    log(f'queue: {todo}')
    return todo


# ---------------- 翻訳（CTranslate2版） ----------------

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
        # チャンクをトークン化して CTranslate2 に渡す
        source_tokens = [
            tok.convert_ids_to_tokens(tok.encode(chunk, truncation=True))
            for chunk in chunks
        ]
        # target_prefix に目標言語コードを指定（NLLBの仕様）
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
            # 先頭の言語コードトークンを除去（NLLBの仕様）
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
    if not book_id.startswith('gutenberg_'):
        log(f'  unknown source: {book_id}')
        return False
    gid = book_id.replace('gutenberg_', '')
    log(f'  fetching text: {book_id}')
    try:
        r = subprocess.run(
            [sys.executable, 'scripts/fetch_gutenberg.py', gid],
            capture_output=True, text=True, timeout=180
        )
        if r.returncode != 0:
            log(f'  fetch failed: {r.stderr.strip()[:300]}')
            return False
        return input_path.exists()
    except Exception as e:
        log(f'  fetch exception: {e}')
        return False


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
        compute_type='int8',       # CPU向けint8量子化
        inter_threads=1,
        intra_threads=1,
    )
    return tok, translator


def main():
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
