"""Knowledge base retrieval module for VoiceOps 2026 ZentroMart support agent.

Chunks Markdown files by [SECTION-ID], preserves metadata (status, effective_from),
enforces document precedence (policy > agent_guide > help > faq > marketing),
caches vector index to disk, and retrieves relevant chunks using cosine similarity.
"""

from pathlib import Path
import csv
import os
import pickle
import re
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT_DIR = Path(__file__).resolve().parent.parent
KB_DIR = ROOT_DIR / "kb"
DATA_DIR = ROOT_DIR / "data"
CACHE_FILE = ROOT_DIR / ".kb_cache.pkl"

PRECEDENCE = {
    "policy": 1,
    "agent_guide": 2,
    "help": 3,
    "faq": 4,
    "marketing": 5,
    "meta": 6,
}


def parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """Parse YAML-like frontmatter from markdown text."""
    meta = {}
    body = text
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2]
            for line in fm_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip().strip('"').strip("'")
    return meta, body


class Chunk:
    def __init__(
        self,
        section_id: str,
        title: str,
        text: str,
        doc_id: str,
        doc_type: str,
        status: str,
        effective_from: str,
        file_name: str,
        rank: int,
    ):
        self.section_id = section_id
        self.title = title
        self.text = text
        self.doc_id = doc_id
        self.doc_type = doc_type
        self.status = status
        self.effective_from = effective_from
        self.file_name = file_name
        self.rank = rank

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "text": self.text,
            "doc_id": self.doc_id,
            "doc_type": self.doc_type,
            "status": self.status,
            "effective_from": self.effective_from,
            "file_name": self.file_name,
            "rank": self.rank,
        }


class KBIndex:
    def __init__(self, use_cache: bool = True):
        self.chunks: List[Chunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.matrix = None
        self.chunk_by_id: Dict[str, Chunk] = {}
        self.products: Dict[str, dict] = {}
        self._load_products()

        if use_cache and CACHE_FILE.exists():
            try:
                self._load_cache()
                return
            except Exception as e:
                pass  # rebuild if cache invalid

        self._build_index()
        self._save_cache()

    def _load_products(self):
        prod_path = DATA_DIR / "products.csv"
        if prod_path.exists():
            with open(prod_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.products[row["sku"]] = row

    def _build_index(self):
        self.chunks = []
        self.chunk_by_id = {}

        # 1. Parse all 21 KB markdown files
        for fpath in sorted(KB_DIR.glob("*.md")):
            raw_text = fpath.read_text(encoding="utf-8")
            meta, body = parse_frontmatter(raw_text)

            doc_id = meta.get("doc_id", fpath.stem)
            doc_type = meta.get("doc_type", "help")
            status = meta.get("status", "current").lower()
            effective_from = meta.get("effective_from", "")

            # Rank: current policy > agent guide > help > FAQ > marketing > meta. Superseded gets high rank (low priority)
            if status == "superseded" or "superseded" in fpath.name:
                rank = 99
            else:
                rank = PRECEDENCE.get(doc_type, 10)

            # Split by ## [SECTION-ID] Title
            sections = re.split(r"(^##\s*\[[^\]]+\][^\n]*)", body, flags=re.M)
            # sections alternate between header and content
            i = 1
            while i < len(sections):
                header = sections[i].strip()
                content = sections[i + 1].strip() if i + 1 < len(sections) else ""
                match = re.search(r"^##\s*\[([^\]]+)\]\s*(.*)$", header)
                if match:
                    sec_id = match.group(1).strip()
                    sec_title = match.group(2).strip()
                    full_text = f"{header}\n{content}"
                    chunk = Chunk(
                        section_id=sec_id,
                        title=sec_title,
                        text=full_text,
                        doc_id=doc_id,
                        doc_type=doc_type,
                        status=status,
                        effective_from=effective_from,
                        file_name=fpath.name,
                        rank=rank,
                    )
                    self.chunks.append(chunk)
                    self.chunk_by_id[sec_id] = chunk
                i += 2

        # 2. Add product entries as searchable chunks: products.csv:{sku}
        for sku, p in self.products.items():
            assured_str = "ZentroMart Assured" if p.get("assured") == "True" else "Standard Seller"
            prod_text = (
                f"Product SKU {sku}: {p.get('title')}. Return category: {p.get('return_category')}. "
                f"Price: Rs. {p.get('price_inr')}. Fulfilled by: {p.get('fulfilled_by')}. "
                f"Assured status: {assured_str}. Warranty: {p.get('warranty_months')} months."
            )
            p_chunk = Chunk(
                section_id=f"products.csv:{sku}",
                title=f"Product {sku}",
                text=prod_text,
                doc_id="products.csv",
                doc_type="product",
                status="current",
                effective_from="2026-08-01",
                file_name="products.csv",
                rank=1,
            )
            self.chunks.append(p_chunk)
            self.chunk_by_id[p_chunk.section_id] = p_chunk

        # 3. Fit TF-IDF Vectorizer
        corpus = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            token_pattern=r"(?u)\b[\w-]+\b",
            lowercase=True,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def _save_cache(self):
        try:
            with open(CACHE_FILE, "wb") as f:
                pickle.dump(
                    {
                        "chunks": [c.to_dict() for c in self.chunks],
                        "vectorizer": self.vectorizer,
                        "matrix": self.matrix,
                    },
                    f,
                )
        except Exception:
            pass

    def _load_cache(self):
        with open(CACHE_FILE, "rb") as f:
            data = pickle.load(f)
            self.vectorizer = data["vectorizer"]
            self.matrix = data["matrix"]
            self.chunks = []
            self.chunk_by_id = {}
            for d in data["chunks"]:
                c = Chunk(**d)
                self.chunks.append(c)
                self.chunk_by_id[c.section_id] = c

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        min_threshold: float = 0.12,
    ) -> Tuple[List[Chunk], List[str], bool]:
        """Retrieve most relevant chunks for a query.

        Returns:
            chunks: List of retrieved Chunk objects
            chunk_ids: List of chunk IDs (e.g., ['RET-2', 'REF-1'])
            is_thin: True if evidence is weak / below threshold
        """
        if not query or not query.strip():
            return [], [], True

        # Check for explicit SKU references (e.g., ELE-0012, CLO-0005)
        sku_matches = re.findall(r"\b([A-Z]{3}-\d{4})\b", query.upper())
        matched_sku_chunks = []
        for sku in sku_matches:
            prod_id = f"products.csv:{sku}"
            if prod_id in self.chunk_by_id:
                matched_sku_chunks.append(self.chunk_by_id[prod_id])

        # Vector search
        q_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self.matrix)[0]

        # Apply precedence adjustments
        # Lower rank (1 = policy) gets a slight boost; superseded gets penalised
        adjusted_scores = []
        for idx, score in enumerate(scores):
            chunk = self.chunks[idx]
            adj = score
            if chunk.rank == 1:
                adj += 0.04
            elif chunk.rank == 2:
                adj += 0.02
            elif chunk.rank >= 99:
                adj -= 0.25  # superseded penalized heavily unless query specifically asks for v3

            adjusted_scores.append((adj, score, idx))

        adjusted_scores.sort(key=lambda x: x[0], reverse=True)

        # Collect top candidates
        retrieved: List[Chunk] = []
        seen_ids = set()

        # Add explicitly matched product chunks first
        for pc in matched_sku_chunks:
            retrieved.append(pc)
            seen_ids.add(pc.section_id)

        # Filter and select top_k KB chunks
        best_raw_score = 0.0
        for adj, raw_score, idx in adjusted_scores:
            chunk = self.chunks[idx]
            if chunk.section_id in seen_ids:
                continue
            if chunk.doc_type == "product":
                continue  # Products handled via explicit/targeted match
            if raw_score > best_raw_score:
                best_raw_score = raw_score

            # Hard negative suppression:
            # If REF-2 is present, suppress FAQ-1 (superseded wallet timeline FAQ)
            if chunk.section_id == "FAQ-1" and any(c.section_id == "REF-2" for c in retrieved):
                continue
            # If RET-2 / RET-3 (v4) is present, suppress RETv3 chunks
            if "RETv3" in chunk.section_id and any("RET-" in c.section_id for c in retrieved):
                continue

            retrieved.append(chunk)
            seen_ids.add(chunk.section_id)
            if len(retrieved) >= top_k:
                break

        # Check if the query is thin (KB has no good match)
        # Note: If SKU matched or best score >= min_threshold, not thin
        is_thin = (best_raw_score < min_threshold) and (len(matched_sku_chunks) == 0)

        chunk_ids = [c.section_id for c in retrieved]
        return retrieved, chunk_ids, is_thin


# Global singleton instance for fast reuse
_KB_INDEX: Optional[KBIndex] = None


def get_kb_index() -> KBIndex:
    global _KB_INDEX
    if _KB_INDEX is None:
        _KB_INDEX = KBIndex(use_cache=True)
    return _KB_INDEX
