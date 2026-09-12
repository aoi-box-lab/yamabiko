#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

TRANSLATIONS_DIR = Path('translations')
MANIFEST_PATH = TRANSLATIONS_DIR / 'manifest.json'


def build():
    books = {}
    for p in TRANSLATIONS_DIR.glob('*.json'):
        if p.name == 'manifest.json':
            continue
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            if not data.get('bookId'):
                continue
            books[data['bookId']] = {
                'size': p.stat().st_size,
                'translator': data.get('translator', 'unknown'),
                'formatVersion': data.get('formatVersion', 0)
            }
        except Exception as e:
            print(f'  skip {p.name}: {e}')

    manifest = {
        'version': 3,
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'books': books
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'manifest: {len(books)} books')


if __name__ == '__main__':
    build()
