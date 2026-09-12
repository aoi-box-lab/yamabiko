#!/usr/bin/env python3
import json
import re
import sys
import urllib.request
from pathlib import Path

def fetch_book(book_id):
    meta_url = f'https://gutendex.com/books/{book_id}'
    try:
        meta = json.loads(urllib.request.urlopen(meta_url, timeout=30).read())
    except Exception as e:
        print(f'  meta failed: {e}')
        return None

    text_url = None
    for fmt in ['text/plain; charset=utf-8', 'text/plain', 'text/html; charset=utf-8']:
        if fmt in meta.get('formats', {}):
            text_url = meta['formats'][fmt]
            break

    if not text_url:
        print(f'  no text format')
        return None

    try:
        raw = urllib.request.urlopen(text_url, timeout=60).read()
        text = raw.decode('utf-8', errors='replace')
    except Exception as e:
        print(f'  fetch failed: {e}')
        return None

    si = text.find('*** START OF')
    ei = text.find('*** END OF')
    if si > 0 and ei > 0:
        text = text[text.find('\n', si):ei]

    if '<html' in text.lower() or '<body' in text.lower():
        text = re.sub(r'<[^>]+>', '', text)

    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 1]

    return {
        'id': f'gutenberg_{book_id}',
        'title': meta.get('title', ''),
        'author': meta.get('authors', [{}])[0].get('name', ''),
        'sourceLang': 'en',
        'paragraphs': paragraphs
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python fetch_gutenberg.py <book_id>')
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
