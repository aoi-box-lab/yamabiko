#!/usr/bin/env python3
import json
import urllib.request
from pathlib import Path

QUEUE_FILE = Path('translate-queue.txt')
TRANSLATIONS_DIR = Path('translations')

QUEUE_MAX = 1000
POPULAR_FETCH_LIMIT = 5000
TARGET_LANGS = {'en'}
USER_AGENT = 'Yamabiko/1.0 (https://github.com/aoi-box-lab/yamabiko)'


def fetch_popular_ids(limit):
    ids = []
    url = 'https://gutendex.com/books?sort=popular'
    page = 0
    while url and len(ids) < limit:
        page += 1
        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': USER_AGENT
            })
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f'  fetch error (page {page}): {e}')
            break

        for b in data.get('results', []):
            langs = set(b.get('languages', []))
            if not (langs & TARGET_LANGS):
                continue
            formats = b.get('formats', {})
            has_text = any(
                k.startswith('text/plain') or k.startswith('text/html')
                for k in formats.keys()
            )
            if not has_text:
                continue
            ids.append(b['id'])

        print(f'  page {page}: total {len(ids)}')
        url = data.get('next')

    return ids


def load_done_ids():
    done = set()
    if TRANSLATIONS_DIR.exists():
        for p in TRANSLATIONS_DIR.glob('*.json'):
            if p.name == 'manifest.json':
                continue
            done.add(p.stem)
    return done


def load_existing_queue():
    if not QUEUE_FILE.exists():
        return []
    return [l.strip() for l in QUEUE_FILE.read_text().split('\n') if l.strip()]


def build_queue():
    print('[update_queue] fetching popular list...')
    popular = fetch_popular_ids(POPULAR_FETCH_LIMIT)
    print(f'[update_queue] fetched: {len(popular)}')

    done = load_done_ids()
    print(f'[update_queue] already translated: {len(done)}')

    existing = load_existing_queue()
    existing_set = set(existing)

    queue = [q for q in existing if q not in done]

    added = 0
    for gid in popular:
        book_id = f'gutenberg_{gid}'
        if book_id in done or book_id in existing_set:
            continue
        queue.append(book_id)
        existing_set.add(book_id)
        added += 1
        if len(queue) >= QUEUE_MAX:
            break

    queue = queue[:QUEUE_MAX]

    QUEUE_FILE.write_text('\n'.join(queue) + '\n', encoding='utf-8')
    print(f'[update_queue] queue updated: {len(queue)} (new: {added})')


if __name__ == '__main__':
    build_queue()
