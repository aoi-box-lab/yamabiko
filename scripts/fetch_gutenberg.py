#!/usr/bin/env python3
import json
import re
import sys
import urllib.request
from pathlib import Path

USER_AGENT = 'Yamabiko/1.0 (https://github.com/aoi-box-lab/yamabiko)'


def fetch_book(book_id):
    meta_url = f'https://gutendex.com/books/{book_id}'
    try:
        req = urllib.request.Request(meta_url, headers={'User-Agent': USER_AGENT})
        meta = json.loads(urllib.request.urlopen(req, timeout=30).read())
    except Exception as e:
        print(f'  meta failed: {e}', file=sys.stderr)
        return None

    text_url = None
    for fmt in ['text/plain; charset=utf-8', 'text/plain', 'text/html; charset=utf-8']:
        if fmt in meta.get('formats', {}):
            text_url = meta['formats'][fmt]
            break

    if not text_url:
        print('  no text format', file=sys.stderr)
        return None

    try:
        req2 = urllib.request.Request(text_url, headers={'User-Agent': USER_AGENT})
        raw = urllib.request.urlopen(req2, timeout=60).read()
        text = raw.decode('utf-8', errors='replace')
    except Exception as e:
        print(f'  fetch failed: {e}', file=sys.stderr)
        return None

    # 改行コード統一
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # START/END マーカーで本文だけ切り出し
    si = text.find('*** START OF')
    ei = text.find('*** END OF')
    if si > 0:
        text = text[text.find('\n', si) + 1:]
    if ei > 0:
        text = text[:ei]

    # HTMLタグ除去
    if '<html' in text.lower() or '<body' in text.lower():
        text = re.sub(r'<[^>]+>', '', text)

    # 段落分割: 空行1つ以上で区切る
    paragraphs = []
    for p in re.split(r'\n\s*\n', text):
        p = p.strip()
        if len(p) < 2:
            continue
        # 段落内の改行・連続空白を1つの空白に
        p = re.sub(r'\s+', ' ', p).strip()
        if len(p) > 1:
            paragraphs.append(p)

    return {
        'id': f'gutenberg_{book_id}',
        'title': meta.get('title', ''),
        'author': meta.get('authors', [{}])[0].get('name', ''),
        'sourceLang': 'en',
        'paragraphs': paragraphs
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
