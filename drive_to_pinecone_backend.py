from flask import Flask, request, jsonify
import requests
import openai
import os
import subprocess
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

app = Flask(__name__)

# === CONFIGURATION ===
HF_TOKEN = "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"
PINECONE_API_KEY = "pcsk_475ix6_QNMj2etqYWbrUz2aKFQebCPzCepmZEsZFoWsMG3wjYvFaxdUFu73h7GWbieTeti"
PINECONE_ENV = "us-east-1"
PINECONE_INDEX = "youtube-transcript"

openai.api_key = "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"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
embedder = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

@app.route("/transcribe", methods=["POSTS"])
def transcribe_drive():
    try:
        data = request.get_json()
        drive_url = data.get("drive_url")

        if not drive_url:
            return jsonify({"success": False, "error": "No URL provided"}), 400

        file_id = extract_drive_file_id(drive_url)
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        filename = "video.mp4"

        print("Downloading video...")
        with requests.get(download_url, stream=True) as r:
            r.raise_for_status()
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        wav_file = "audio.wav"
        print("Converting to audio.wav...")
        subprocess.run(["ffmpeg", "-y", "-i", filename, wav_file], check=True)

        print("Transcribing via Hugging Face Whisper...")
        with open(wav_file, "rb") as f:
            audio_data = f.read()

        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        response = requests.post(
            "https://api-inference.huggingface.co/models/openai/whisper-base",
            headers=headers,
            data=audio_data
        )

        if response.status_code != 200:
            return jsonify({"success": False, "error": response.text}), 500

        result = response.json()
        full_text = result.get("text", "")

        print("Chunking and embedding...")
        chunks = chunk_text(full_text)
        vectors = embedder.encode(chunks).tolist()

        print("Uploading to Pinecone...")
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            index.upsert([
                {
                    "id": f"{file_id}_chunk_{i}",
                    "values": vector,
                    "metadata": {
                        "video_id": file_id,
                        "chunk_index": i,
                        "text": chunk
                    }
                }
            ])

        os.remove(filename)
        os.remove(wav_file)

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

def extract_drive_file_id(url):
    if "id=" in url:
        return url.split("id=")[-1].split("&")[0]
    elif "/d/" in url:
        return url.split("/d/")[1].split("/")[0]
    else:
        raise ValueError("Invalid Google Drive URL")

def chunk_text(text, max_len=500):
    words = text.split()
    chunks = []
    current = []
    total = 0
    for word in words:
        if total + len(word) + 1 > max_len:
            chunks.append(" ".join(current))
            current = [word]
            total = len(word) + 1
        else:
            current.append(word)
            total += len(word) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks

if __name__ == "__main__":
    port = int(os.environ.get("PORTS", 8080))
    print(f"✅ Starting Flask app on port {port}...")
    app.run(host="0.0.0.0", port=port)
