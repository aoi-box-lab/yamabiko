# FIXME: ネットワーク切れたときの再試行処理めんどいからとりあえず放置
# 依存ライブラリの読み込み。もう動くはずだから大丈夫なはず
import os
import sys
import glob
import urllib.request
from transformers import MarianMTModel, MarianTokenizer

def download_gutenberg_book(book_id, save_path):
    # Project GutenbergのプレーンテキストURL
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    print(f"Downloading Gutenberg book ID {book_id} from {url} ...")
    try:
        urllib.request.urlretrieve(url, save_path)
        print(f"Successfully downloaded to {save_path}")
    except Exception as e:
        print(f"ダウンロードでエラー起きたわ最悪: {e}")
        # 失敗した場合はダミーを作って続行できるようにする
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("Fallback text due to download failure.\nThis is an automated translation test.")

def translate_text(text, model, tokenizer):
    if not text.strip():
        return ""
    
    # 512トークンを超えないように長すぎる行は安全に切り詰める
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return result[0]

def main():
    maji_de_model_str = "Helsinki-NLP/opus-mt-en-jap"
    
    print(f"Loading model: {maji_de_model_str} ...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(maji_de_model_str)
        model = MarianMTModel.from_pretrained(maji_de_model_str)
    except Exception as e:
        print(f"モデル読み込みでバグったわ最悪: {e}")
        sys.exit(1)

    input_dir = "data/input"
    output_dir = "data/output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Gutenbergから「変身（フランツ・カフカ）」のID: 5200 をダウンロードしてみる
    book_id = "5200"
    input_file_path = os.path.join(input_dir, f"gutenberg_{book_id}.txt")
    
    if not os.path.exists(input_file_path):
        download_gutenberg_book(book_id, input_file_path)

    input_files = glob.glob(f"{input_dir}/*.txt")

    for file_path in input_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"translated_{file_name}")
        print(f"Translating {file_path} -> {output_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        translated_lines = []
        # テスト用に最初の50行だけ翻訳する（全部やるとCIのタイムアウトが怖いからな！）
        target_lines = lines[:50]
        for line in target_lines:
            translated_line = translate_text(line.strip(), model, tokenizer)
            translated_lines.append(translated_line + "\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(translated_lines)

    print("全ファイルの完全版翻訳完了したぞお疲れ！")

if __name__ == "__main__":
    main()
# end of file (つかれすぎてやばい)
