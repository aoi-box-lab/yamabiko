#!/usr/bin/env python3
import json
import re
import sys
import urllib.request
from pathlib import Path

USER_AGENT = 'Yamabiko/1.0 (https://github.com/aoi-box-lab/yamabiko)'

GUTENBERG_MIRRORS = [
    'https://www.gutenberg.org',
    'https://gutenberg.pglaf.org',
    'https://mirrors.xmission.com/gutenberg',
]


def fetch_book(book_id):
    text = None
    for base in GUTENBERG_MIRRORS:
        url = f'{base}/cache/epub/{book_id}/pg{book_id}.txt'
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                text = r.read().decode('utf-8', errors='replace')
            print(f'  fetched from {base}')
            break
        except Exception as e:
            print(f'  {base} failed: {e}', file=sys.stderr)
            continue

    if not text:
        print('  all mirrors failed', file=sys.stderr)
        return None

    text = text.replace('\r\n', '\n').replace('\r', '\n')

    si = text.find('*** START OF')
    ei = text.find('*** END OF')
    if si > 0:
        text = text[text.find('\n', si) + 1:]
    if ei > 0:
        text = text[:ei]

    if '<html' in text.lower() or '<body' in text.lower():
        text = re.sub(r'<[^>]+>', '', text)

    paragraphs = []
    for p in re.split(r'\n\s*\n', text):
        p = p.strip()
        if len(p) < 2:
            continue
        p = re.sub(r'\s+', ' ', p).strip()
        if len(p) > 1:
            paragraphs.append(p)

    return {
        'id': f'gutenberg_{book_id}',
        'title': '',
        'author': '',
        'sourceLang': 'en',
        'paragraphs': paragraphs,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python fetch_gutenberg.py <book_id>', file=sys.stderr)
        sys.exit(1)

    book_id = sys.argv[1]
    out_dir = Path('input')
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f'gutenberg_{book_id}.json'

    if out_path.exists():
        print(f'skip (cached): {out_path}')
        sys.exit(0)

    book = fetch_book(book_id)
    if not book:
        sys.exit(1)

    out_path.write_text(json.dumps(book, ensure_ascii=False), encoding='utf-8')
    print(f'saved: {out_path} ({len(book["paragraphs"])} paras)')
