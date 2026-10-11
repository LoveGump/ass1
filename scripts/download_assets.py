"""Download pinned public assets; proxy/mirror settings come from the environment."""
import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
GLOVE_MD5 = "40ec481866001177b8cd4cb0df92924f"


def download(url, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    partial = target.with_suffix(target.suffix + ".part")
    subprocess.run(["curl.exe" if os.name == "nt" else "curl", "-L", "--fail", "--retry", "3", "--connect-timeout", "30",
                    "--silent", "--show-error", "-C", "-", "-o", str(partial), url], check=True)
    partial.replace(target)
    print(f"Downloaded {target.name}: {target.stat().st_size} bytes", flush=True)


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["glove", "bert", "all"], default="all")
    parser.add_argument("--glove-source", choices=["stanford", "gensim"], default="stanford")
    args = parser.parse_args()
    directory = ROOT / "artifacts"
    directory.mkdir(exist_ok=True)
    manifest_path = ROOT / "results" / "assets.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.only in ("all", "glove") and args.glove_source == "gensim":
        url = "https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-wiki-gigaword-100/glove-wiki-gigaword-100.gz"
        path = directory / "glove-wiki-gigaword-100.gz"
        download(url, path)
        if digest(path, "md5") != GLOVE_MD5:
            raise ValueError("GloVe checksum mismatch; remove corrupt file before retrying.")
        manifest["glove"] = {"url": url, "md5": GLOVE_MD5, "sha256": digest(path, "sha256"),
                             "original": "https://nlp.stanford.edu/data/glove.6B.zip",
                             "format": "Lossless Gensim word2vec-text conversion of glove.6B.100d, 400000 vectors"}
    elif args.only in ("all", "glove"):
        url = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
        archive = directory / "glove.6B.zip"
        target = directory / "glove.6B.100d.txt"
        download(url, archive)
        if archive.stat().st_size != 862182613:
            raise ValueError("Incomplete Stanford archive; rename to glove.6B.zip.part to resume.")
        with zipfile.ZipFile(archive) as z:
            if not target.exists():
                z.extract("glove.6B.100d.txt", directory)
        manifest["glove"] = {"url": url, "archive_sha256": digest(archive, "sha256"),
                             "sha256": digest(target, "sha256"),
                             "format": "Original glove.6B.100d.txt, 400000 vectors"}
    if args.only in ("all", "bert"):
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        api_url = f"{endpoint}/api/models/google-bert/bert-base-uncased/revision/{REVISION}?blobs=true"
        with urlopen(api_url, timeout=60) as response:
            info = json.load(response)
        siblings = {f["rfilename"]: f for f in info["siblings"]}
        files = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"]
        hashes = {}
        for name in files:
            path = directory / "bert-base-uncased" / name
            download(f"{endpoint}/google-bert/bert-base-uncased/resolve/{REVISION}/{name}", path)
            actual = digest(path, "sha256")
            expected = siblings[name].get("lfs", {}).get("sha256")
            if expected and actual != expected:
                raise ValueError(f"Checksum mismatch: {name}")
            hashes[name] = actual
        manifest["bert"] = {"model_id": "google-bert/bert-base-uncased", "revision": REVISION,
                            "endpoint": endpoint, "sha256": hashes}
    manifest_path.parent.mkdir(exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
