from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping, Sequence

import torch


class FingerprintRetriever:
    def __init__(
        self,
        top_k: int = 4,
        sim_th: float = 0.35,
        feature_dim: int = 64,
        simhash_bits: int = 64,
        cache_enabled: bool = True,
        prefilter_factor: int = 8,
    ) -> None:
        self.top_k = max(int(top_k), 1)
        self.sim_th = float(sim_th)
        self.feature_dim = int(feature_dim)
        self.simhash_bits = int(simhash_bits)
        self.cache_enabled = bool(cache_enabled)
        self.prefilter_factor = max(int(prefilter_factor), 1)

        generator = torch.Generator().manual_seed(17)
        self._hyperplanes = torch.randn(
            self.simhash_bits,
            self.feature_dim,
            generator=generator,
            dtype=torch.float32,
        )

        self._index: list[dict[str, Any]] = []
        self._cache: dict[str, dict[str, Any]] = {}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "FingerprintRetriever":
        retriever_cfg = config.get("retriever", {})
        model_cfg = config.get("model", {})
        return cls(
            top_k=int(retriever_cfg.get("top_k", 4)),
            sim_th=float(retriever_cfg.get("sim_th", 0.35)),
            feature_dim=int(model_cfg.get("sample_feature_dim", 64)),
            simhash_bits=int(retriever_cfg.get("simhash_bits", 64)),
            cache_enabled=bool(retriever_cfg.get("cache_enabled", True)),
            prefilter_factor=int(retriever_cfg.get("prefilter_factor", 8)),
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _vectorize_text(self, text: str) -> torch.Tensor:
        vector = torch.zeros(self.feature_dim, dtype=torch.float32)
        tokens = self._tokenize(text)
        if not tokens:
            vector[0] = 1.0
            return vector

        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.feature_dim
            sign = 1.0 if (digest[4] & 1) else -1.0
            weight = 1.0 + (int(digest[5]) / 255.0)
            vector[index] += sign * weight

        norm = torch.linalg.norm(vector)
        if norm.item() == 0.0:
            vector[0] = 1.0
            return vector
        return vector / norm

    def encode_query(self, payload: Mapping[str, Any] | str) -> torch.Tensor:
        if isinstance(payload, str):
            return self._vectorize_text(payload)

        embedding_raw = payload.get("embedding")
        if isinstance(embedding_raw, torch.Tensor):
            embedding = embedding_raw.detach().to(dtype=torch.float32)
            if embedding.dim() != 1 or embedding.shape[0] != self.feature_dim:
                raise ValueError(
                    f"query embedding must be shape ({self.feature_dim},), got {tuple(embedding.shape)}"
                )
            norm = torch.linalg.norm(embedding)
            return embedding if norm.item() == 0.0 else embedding / norm

        instruction = str(payload.get("instruction", ""))
        inp = str(payload.get("input", ""))
        return self._vectorize_text(f"{instruction}\n{inp}".strip())

    def _simhash(self, embedding: torch.Tensor) -> int:
        signs = torch.mv(self._hyperplanes, embedding) >= 0
        value = 0
        for idx, bit in enumerate(signs.tolist()):
            if bit:
                value |= 1 << idx
        return value

    @staticmethod
    def _hamming_similarity(left: int, right: int, num_bits: int) -> float:
        distance = (left ^ right).bit_count()
        return 1.0 - (float(distance) / float(num_bits))

    @staticmethod
    def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
        return float(torch.dot(left, right).item())

    @staticmethod
    def _cache_key(payload: Mapping[str, Any] | str) -> str:
        if isinstance(payload, str):
            attack_family = "unknown"
            key_text = payload
        else:
            attack_family = str(payload.get("attack_family", "unknown"))
            key_text = str(payload.get("instruction", ""))

        key_hash = hashlib.sha1(key_text.encode("utf-8")).hexdigest()
        return f"{attack_family}:{key_hash}"

    @staticmethod
    def _clone_output(output: dict[str, Any]) -> dict[str, Any]:
        cloned: dict[str, Any] = {}
        for key, value in output.items():
            if isinstance(value, torch.Tensor):
                cloned[key] = value.clone()
            elif isinstance(value, list):
                cloned[key] = list(value)
            elif isinstance(value, dict):
                cloned[key] = dict(value)
            else:
                cloned[key] = value
        return cloned

    @property
    def index_size(self) -> int:
        return len(self._index)

    def build_index(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._index.clear()
        self._cache.clear()
        for record in records:
            embedding = self.encode_query(record)
            item = {
                "sample_id": str(record.get("sample_id", f"idx-{len(self._index)}")),
                "attack_family": str(record.get("attack_family", "unknown")),
                "instruction": str(record.get("instruction", "")),
                "embedding": embedding,
                "simhash": self._simhash(embedding),
            }
            self._index.append(item)

    def retrieve(self, query: Mapping[str, Any] | str) -> dict[str, Any]:
        key = self._cache_key(query)
        if self.cache_enabled and key in self._cache:
            cached = self._clone_output(self._cache[key])
            cached["cache_hit"] = True
            return cached

        query_embedding = self.encode_query(query)
        if not self._index:
            result = {
                "topk_ids": [],
                "topk_scores": [],
                "memory_tokens": query_embedding.unsqueeze(0),
                "retriever_stats": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "fallback": True,
                "cache_key": key,
                "cache_hit": False,
                "query_embedding": query_embedding,
            }
            if self.cache_enabled:
                self._cache[key] = self._clone_output(result)
            return result

        query_hash = self._simhash(query_embedding)
        prefilter_k = min(len(self._index), self.top_k * self.prefilter_factor)

        prefiltered = sorted(
            self._index,
            key=lambda item: self._hamming_similarity(
                query_hash,
                int(item["simhash"]),
                self.simhash_bits,
            ),
            reverse=True,
        )[:prefilter_k]

        reranked: list[tuple[float, dict[str, Any]]] = []
        for item in prefiltered:
            score = self._cosine_similarity(query_embedding, item["embedding"])
            reranked.append((score, item))
        reranked.sort(key=lambda pair: pair[0], reverse=True)

        selected = [(score, item) for score, item in reranked if score >= self.sim_th][: self.top_k]
        if not selected:
            result = {
                "topk_ids": [],
                "topk_scores": [],
                "memory_tokens": query_embedding.unsqueeze(0),
                "retriever_stats": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "fallback": True,
                "cache_key": key,
                "cache_hit": False,
                "query_embedding": query_embedding,
            }
            if self.cache_enabled:
                self._cache[key] = self._clone_output(result)
            return result

        topk_ids = [str(item["sample_id"]) for _, item in selected]
        topk_scores = [float(score) for score, _ in selected]
        memory_tokens = torch.stack([item["embedding"] for _, item in selected], dim=0)
        retriever_stats = torch.tensor(
            [max(topk_scores), sum(topk_scores) / len(topk_scores)],
            dtype=torch.float32,
        )
        result = {
            "topk_ids": topk_ids,
            "topk_scores": topk_scores,
            "memory_tokens": memory_tokens,
            "retriever_stats": retriever_stats,
            "fallback": False,
            "cache_key": key,
            "cache_hit": False,
            "query_embedding": query_embedding,
        }
        if self.cache_enabled:
            self._cache[key] = self._clone_output(result)
        return result
