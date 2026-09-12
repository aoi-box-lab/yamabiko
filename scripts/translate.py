# FIXME: あとでリファクタリングする（たぶんやらない）
# 依存ライブラリの読み込み。エラー出たらキレそう
import os
import sys
import glob
from transformers import MarianMTModel, MarianTokenizer

def translate_text(text, model, tokenizer):
    # 空行ならそのまま返すぞ
    if not text.strip():
        return ""
    
    # トークナイズして翻訳を実行する処理
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated = model.generate(**inputs)
    result = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return result[0]

def main():
    # maji_de_model_str = "staka/fugumtx-en-ja"  # 英語から日本語へのモデル
    maji_de_model_str = "Helsinki-NLP/opus-mt-en-jap"
    
    print(f"Loading model: {maji_de_model_str} ...")
    try:
        tokenizer = MarianTokenizer.from_pretrained(maji_de_model_str)
        model = MarianMTModel.from_pretrained(maji_de_model_str)
    except Exception as e:
        print(f"モデル読み込みでバグったわ最悪: {e}")
        sys.exit(1)

    # 入力テキストファイルを探す処理（なければサンプル作成）
    input_dir = "data/input"
    output_dir = "data/output"
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    input_files = glob.glob(f"{input_dir}/*.txt")
    if not input_files:
        print("翻訳対象ファイルがないからダミー作るわ...")
        sample_path = os.path.join(input_dir, "sample.txt")
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write("Hello world!\nThis is an automated translation test.")
        input_files = [sample_path]

    for file_path in input_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"translated_{file_name}")
        print(f"Translating {file_path} -> {output_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        translated_lines = []
        for line in lines:
            # 1行ずつ翻訳処理を回す
            translated_line = translate_text(line, model, tokenizer)
            translated_lines.append(translated_line + "\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.writelines(translated_lines)

    print("全ファイルの翻訳完了したぞお疲れ！")

if __name__ == "__main__":
    main()
# end of file (つかれた)
