import os, json
from config import DATA_DIR

class PersistentMemory:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.file = os.path.join(DATA_DIR, "memory.json")
        if os.path.exists(self.file):
            with open(self.file,"r") as f:
                self.data = json.load(f)
        else:
            self.data = []

    def add(self, texts, metas=None):
        for i,t in enumerate(texts):
            meta = metas[i] if metas and i<len(metas) else {}
            self.data.append({"text":t,"meta":meta})
        self._save()

    def search(self, query, k=3):
        results = [(entry["text"], entry.get("meta",{}), 0) for entry in self.data if query.lower() in entry["text"].lower()]
        return results[:k]

    def _save(self):
        with open(self.file,"w") as f:
            json.dump(self.data,f,indent=2)
