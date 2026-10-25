"""
Export the Locus product catalog from a Qdrant instance to a file.

Writes data/catalog.jsonl.gz: one line per point (id, vector, payload),
preceded by one header line per collection with its vector config.
The demo container loads this file into its own local Qdrant at startup
(scripts/load_catalog.py), so the demo needs no paid database.

Usage (reads QDRANT_URL / QDRANT_API_KEY from the environment or .env):
    python scripts/export_catalog.py
"""
import gzip
import json
import os
import pathlib
import sys

from qdrant_client import QdrantClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

COLLECTIONS = ["locus_items"]
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "catalog.jsonl.gz"


def vector_config(info):
    vectors = info.config.params.vectors
    if hasattr(vectors, "size"):
        return {"size": vectors.size, "distance": str(vectors.distance.value)}
    return {name: {"size": v.size, "distance": str(v.distance.value)} for name, v in vectors.items()}


def main():
    url = os.getenv("QDRANT_URL")
    if not url:
        sys.exit("Set QDRANT_URL and QDRANT_API_KEY first.")
    client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"), timeout=60)
    OUT.parent.mkdir(exist_ok=True)

    with gzip.open(OUT, "wt", encoding="utf-8") as f:
        for name in COLLECTIONS:
            info = client.get_collection(name)
            f.write(json.dumps({"collection": name, "vectors": vector_config(info)}) + "\n")
            offset, count = None, 0
            while True:
                points, offset = client.scroll(
                    collection_name=name, limit=256, offset=offset,
                    with_payload=True, with_vectors=True,
                )
                for p in points:
                    f.write(json.dumps({"c": name, "id": p.id, "v": p.vector, "p": p.payload}) + "\n")
                count += len(points)
                print(f"  {name}: {count} points", end="\r")
                if offset is None:
                    break
            print(f"  {name}: {count} points")
    print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
