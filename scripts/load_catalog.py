"""
Load data/catalog.jsonl.gz into the local Qdrant (run by start.sh before the gateway starts).
"""
import gzip
import json
import os
import pathlib
import time

from qdrant_client import QdrantClient, models

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "catalog.jsonl.gz"
BATCH = 256


def params(cfg):
    if "size" in cfg:
        return models.VectorParams(size=cfg["size"], distance=models.Distance(cfg["distance"]))
    return {k: models.VectorParams(size=v["size"], distance=models.Distance(v["distance"])) for k, v in cfg.items()}


def main():
    client = QdrantClient(host=os.getenv("QDRANT_HOST", "127.0.0.1"),
                          port=int(os.getenv("QDRANT_PORT", 6333)), timeout=60)
    for _ in range(60):
        try:
            client.get_collections()
            break
        except Exception:
            time.sleep(1)

    if not DATA.exists():
        print(f"[catalog] {DATA} not found, starting with an empty catalog")
        return

    batch, totals = [], {}

    def flush():
        if batch:
            client.upsert(collection_name=batch[0][0], points=[b[1] for b in batch], wait=True)
            batch.clear()

    with gzip.open(DATA, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "collection" in row:
                flush()
                name = row["collection"]
                if client.collection_exists(name):
                    client.delete_collection(name)
                client.create_collection(name, vectors_config=params(row["vectors"]))
                continue
            if batch and batch[0][0] != row["c"]:
                flush()
            batch.append((row["c"], models.PointStruct(id=row["id"], vector=row["v"], payload=row["p"])))
            totals[row["c"]] = totals.get(row["c"], 0) + 1
            if len(batch) >= BATCH:
                flush()
    flush()
    for name, n in totals.items():
        print(f"[catalog] loaded {n} points into {name}")


if __name__ == "__main__":
    main()
