"""Shard (``.npz``) format, ``meta.json`` and a torch ``Dataset`` over a directory of shards.

Shard arrays (right-padded to ``L``)::

    tokens        int16   [N, L]     token ids
    action_mask   bool    [N, L]     decision positions
    action_target int8    [N, L]     BR action at decision positions, -1 elsewhere
    legal         bool    [N, L, 3]  legal actions at decision positions (all-False elsewhere)
    belief_mask   bool    [N, L]     BOS and END_HAND positions
    opp_id        int32   [N]        index into the population (-1 for a continuous prior)
    theta         float32 [N, P]     opponent parameters
    archetype     int32   [N]        archetype id

``Shard.load`` refuses dtype mismatches and ``ShardDataset`` validates every shard (against the
tokenizer rebuilt from ``meta.json``) the first time it is loaded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from exsolver.data.records import N_ACTIONS
from exsolver.data.tokenizer import Tokenizer, TokenizerSpec

SHARD_DTYPES: dict[str, type] = {
    "tokens": np.int16,
    "action_mask": np.bool_,
    "action_target": np.int8,
    "legal": np.bool_,
    "belief_mask": np.bool_,
    "opp_id": np.int32,
    "theta": np.float32,
    "archetype": np.int32,
}
SHARD_KEYS: tuple[str, ...] = tuple(SHARD_DTYPES)
META_FILENAME = "meta.json"
SHARD_GLOB = "*.npz"

# torch dtypes handed to the model / losses.
_TORCH_DTYPES: dict[str, torch.dtype] = {
    "tokens": torch.long,
    "action_mask": torch.bool,
    "action_target": torch.long,
    "legal": torch.bool,
    "belief_mask": torch.bool,
    "opp_id": torch.long,
    "theta": torch.float32,
    "archetype": torch.long,
}

# Fallback tokenizer specs when meta.json lacks n_cards / max_result.
_GAME_SPECS: dict[str, tuple[int, int]] = {"kuhn": (3, 2), "leduc": (6, 13)}


@dataclass
class Shard:
    """In-memory shard: one numpy array per format key, validated on request."""

    tokens: np.ndarray
    action_mask: np.ndarray
    action_target: np.ndarray
    legal: np.ndarray
    belief_mask: np.ndarray
    opp_id: np.ndarray
    theta: np.ndarray
    archetype: np.ndarray

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_arrays(cls, **arrays: np.ndarray) -> Shard:
        """Build a shard, explicitly casting each array to its canonical dtype."""
        missing = set(SHARD_KEYS) - set(arrays)
        extra = set(arrays) - set(SHARD_KEYS)
        if missing or extra:
            raise ValueError(f"bad shard keys: missing={sorted(missing)} extra={sorted(extra)}")
        return cls(
            **{k: np.ascontiguousarray(arrays[k], dtype=SHARD_DTYPES[k]) for k in SHARD_KEYS}
        )

    @classmethod
    def empty(cls, n: int, L: int, P: int) -> Shard:
        """All-PAD shard with ``n`` sessions (targets -1, ids -1)."""
        return cls(
            tokens=np.zeros((n, L), np.int16),
            action_mask=np.zeros((n, L), np.bool_),
            action_target=np.full((n, L), -1, np.int8),
            legal=np.zeros((n, L, N_ACTIONS), np.bool_),
            belief_mask=np.zeros((n, L), np.bool_),
            opp_id=np.full(n, -1, np.int32),
            theta=np.zeros((n, P), np.float32),
            archetype=np.zeros(n, np.int32),
        )

    @staticmethod
    def concat(shards: Sequence[Shard]) -> Shard:
        if not shards:
            raise ValueError("cannot concatenate zero shards")
        return Shard(**{k: np.concatenate([getattr(s, k) for s in shards]) for k in SHARD_KEYS})

    # ------------------------------------------------------------------ views
    @property
    def n(self) -> int:
        return int(self.tokens.shape[0])

    @property
    def L(self) -> int:
        return int(self.tokens.shape[1])

    @property
    def P(self) -> int:
        return int(self.theta.shape[1])

    def __len__(self) -> int:
        return self.n

    def as_dict(self) -> dict[str, np.ndarray]:
        return {k: getattr(self, k) for k in SHARD_KEYS}

    def take(self, idx: np.ndarray | Sequence[int] | slice) -> Shard:
        """Row subset (fancy indexing along the session axis)."""
        return Shard(**{k: getattr(self, k)[idx] for k in SHARD_KEYS})

    def iter_chunks(self, size: int) -> Iterator[Shard]:
        for start in range(0, self.n, size):
            yield self.take(slice(start, start + size))

    # ------------------------------------------------------------------ validation
    def check_dtypes(self) -> None:
        """Raise ``ValueError`` unless every array has exactly its canonical dtype."""
        for f in fields(self):
            arr = getattr(self, f.name)
            want = np.dtype(SHARD_DTYPES[f.name])
            if not isinstance(arr, np.ndarray) or arr.dtype != want:
                raise ValueError(
                    f"{f.name}: expected dtype {want}, got {getattr(arr, 'dtype', type(arr))}"
                )

    def validate(
        self,
        L: int | None = None,
        vocab_size: int | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Raise ``ValueError`` on any dtype / shape / consistency violation.

        Always checked: dtypes, shapes, ``action_target == -1`` off-mask and legal on-mask,
        ``legal`` all-False off-mask with at least one legal action on-mask.  With a
        ``tokenizer``: ``action_mask`` equals the positions followed by ``ME_*`` tokens,
        ``belief_mask`` equals the BOS / END_HAND positions, sessions start with BOS and all
        tokens are in the vocabulary.
        """
        self.check_dtypes()
        n, seq = self.tokens.shape
        if L is not None and seq != L:
            raise ValueError(f"tokens: expected L={L}, got {seq}")
        for key in ("action_mask", "action_target", "belief_mask"):
            if getattr(self, key).shape != (n, seq):
                raise ValueError(
                    f"{key}: expected shape {(n, seq)}, got {getattr(self, key).shape}"
                )
        if self.legal.shape != (n, seq, N_ACTIONS):
            raise ValueError(f"legal: expected shape {(n, seq, N_ACTIONS)}, got {self.legal.shape}")
        for key in ("opp_id", "archetype"):
            if getattr(self, key).shape != (n,):
                raise ValueError(f"{key}: expected shape {(n,)}, got {getattr(self, key).shape}")
        if self.theta.ndim != 2 or self.theta.shape[0] != n:
            raise ValueError(f"theta: expected shape ({n}, P), got {self.theta.shape}")
        if tokenizer is not None and vocab_size is None:
            vocab_size = tokenizer.vocab_size
        if (
            vocab_size is not None
            and n
            and (self.tokens.min() < 0 or self.tokens.max() >= vocab_size)
        ):
            raise ValueError("tokens out of vocabulary range")
        if np.any(self.action_target[~self.action_mask] != -1):
            raise ValueError("action_target must be -1 off the action mask")
        on = self.action_target[self.action_mask]
        if on.size and (on.min() < 0 or on.max() >= N_ACTIONS):
            raise ValueError("action_target at decision positions must be in [0, 3)")
        if self.legal[~self.action_mask].any():
            raise ValueError("legal must be all-False off the action mask")
        legal_on = self.legal[self.action_mask]
        if legal_on.size and not legal_on.any(axis=1).all():
            raise ValueError("every decision position needs at least one legal action")
        if on.size and not legal_on[np.arange(on.size), on.astype(np.int64)].all():
            raise ValueError("action_target must be legal at every decision position")
        if tokenizer is not None:
            if n and not (self.tokens[:, 0] == tokenizer.BOS).all():
                raise ValueError("every session must start with BOS")
            if not np.array_equal(self.action_mask, tokenizer.decision_mask(self.tokens)):
                raise ValueError("action_mask is not the set of positions followed by ME_* tokens")
            if not np.array_equal(self.belief_mask, tokenizer.belief_mask(self.tokens)):
                raise ValueError("belief_mask is not the set of BOS / END_HAND positions")

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path, compress: bool = False) -> Path:
        self.check_dtypes()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saver = np.savez_compressed if compress else np.savez
        saver(path, **self.as_dict())
        return path

    @classmethod
    def load(cls, path: str | Path) -> Shard:
        """Load a shard; raises ``ValueError`` on missing keys or any dtype mismatch (no casting)."""
        with np.load(path) as npz:
            missing = set(SHARD_KEYS) - set(npz.files)
            if missing:
                raise ValueError(f"{path}: missing arrays {sorted(missing)}")
            arrays = {k: npz[k] for k in SHARD_KEYS}
        for key, arr in arrays.items():
            want = np.dtype(SHARD_DTYPES[key])
            if arr.dtype != want:
                raise ValueError(f"{path}: {key} has dtype {arr.dtype}, expected {want}")
        return cls(**arrays)


# ---------------------------------------------------------------------- directory-level helpers
def shard_paths(data_dir: str | Path) -> list[Path]:
    """Sorted shard files in ``data_dir``."""
    return sorted(Path(data_dir).glob(SHARD_GLOB))


def write_shards(
    out_dir: str | Path,
    shards: Shard | Iterable[Shard],
    shard_size: int | None = None,
    prefix: str = "shard",
    compress: bool = False,
    start_index: int = 0,
) -> list[Path]:
    """Write one or more shards as ``{prefix}_{i:05d}.npz``; re-chunk to ``shard_size`` if given."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(shards, Shard):
        shards = [shards]
    paths: list[Path] = []
    i = start_index
    for shard in shards:
        chunks = shard.iter_chunks(shard_size) if shard_size else [shard]
        for chunk in chunks:
            if chunk.n == 0:
                continue
            paths.append(chunk.save(out_dir / f"{prefix}_{i:05d}.npz", compress=compress))
            i += 1
    return paths


def load_all(data_dir: str | Path) -> Shard:
    """Concatenate every shard in ``data_dir`` into one in-memory ``Shard`` (no validation)."""
    paths = shard_paths(data_dir)
    if not paths:
        raise FileNotFoundError(f"no {SHARD_GLOB} shards in {data_dir}")
    return Shard.concat([Shard.load(p) for p in paths])


def write_meta(data_dir: str | Path, meta: dict[str, Any]) -> Path:
    """Write the free-form ``meta.json`` next to the shards."""
    path = Path(data_dir) / META_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=_json_default) + "\n")
    return path


def read_meta(data_dir: str | Path) -> dict[str, Any]:
    """Read ``meta.json``; ``{}`` if absent."""
    path = Path(data_dir) / META_FILENAME
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def make_meta(
    game: str,
    vocab_size: int,
    L: int,
    H: int,
    population: str | dict[str, Any] | None = None,
    collection_mix: dict[str, float] | None = None,
    seed: int | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Assemble the standard meta dict (any extra keys are passed through)."""
    meta: dict[str, Any] = {
        "game": game,
        "vocab_size": int(vocab_size),
        "L": int(L),
        "H": int(H),
        "population": population,
        "collection_mix": collection_mix,
        "seed": seed,
    }
    meta.update(extra)
    return meta


def tokenizer_spec_from_meta(meta: dict[str, Any]) -> TokenizerSpec | None:
    """``TokenizerSpec`` from ``meta.json`` (``n_cards``/``max_result``, else by game name)."""
    if "n_cards" in meta and "max_result" in meta:
        return TokenizerSpec(int(meta["n_cards"]), int(meta["max_result"]))
    game = str(meta.get("game", "")).lower()
    for key, (n, m) in _GAME_SPECS.items():
        if game.startswith(key):
            return TokenizerSpec(n, m)
    return None


def tokenizer_from_meta(meta: dict[str, Any]) -> Tokenizer | None:
    """Tokenizer rebuilt from ``meta.json``, or ``None`` when the meta does not identify one."""
    spec = tokenizer_spec_from_meta(meta)
    return None if spec is None else Tokenizer(spec)


def file_hash(path: str | Path, algo: str = "sha256") -> str:
    """Hex digest of a file (for recording the population file in ``meta.json``)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer | np.floating | np.bool_):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


# ---------------------------------------------------------------------- torch dataset
def to_tensors(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    """Convert shard arrays (with or without a leading batch dim) to model-ready tensors."""
    # np.asarray keeps 0-d scalars 0-d (ascontiguousarray would promote them to shape [1]).
    return {k: torch.from_numpy(np.asarray(v)).to(_TORCH_DTYPES[k]) for k, v in arrays.items()}


def collate(samples: Sequence[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of ``ShardDataset`` samples into one batch dict."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


ShardValidator = Callable[[Shard, Path], None]


class _ShardStore:
    """Lazy per-shard cache shared by a dataset and its subsets.

    Reads only the tiny ``opp_id`` array of every shard at construction (row counts and ids);
    full shards are loaded on first access, validated once, and cached.
    """

    def __init__(
        self, paths: Sequence[Path], max_cached: int | None, validator: ShardValidator | None
    ) -> None:
        if not paths:
            raise FileNotFoundError("no shard files")
        self.paths = list(paths)
        self.opp_ids: list[np.ndarray] = [_read_opp_ids(p) for p in self.paths]
        self.sizes = np.array([len(o) for o in self.opp_ids], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])
        self.total = int(self.offsets[-1])
        self.max_cached = max_cached
        self.validator = validator
        self._cache: dict[int, Shard] = {}
        self._validated: set[int] = set()

    def shard(self, k: int) -> Shard:
        s = self._cache.get(k)
        if s is None:
            if self.max_cached is not None and len(self._cache) >= self.max_cached:
                self._cache.pop(next(iter(self._cache)))
            s = Shard.load(self.paths[k])
            if self.validator is not None and k not in self._validated:
                self.validator(s, self.paths[k])
                self._validated.add(k)
            self._cache[k] = s
        return s

    def locate(self, global_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        k = np.searchsorted(self.offsets, global_idx, side="right") - 1
        return k, global_idx - self.offsets[k]

    def all_opp_ids(self) -> np.ndarray:
        return np.concatenate(self.opp_ids) if self.opp_ids else np.zeros(0, np.int32)


def _read_opp_ids(path: Path) -> np.ndarray:
    with np.load(path) as npz:
        if "opp_id" not in npz.files:
            raise ValueError(f"{path}: missing array 'opp_id'")
        return np.asarray(npz["opp_id"])


def _make_validator(meta: dict[str, Any]) -> ShardValidator:
    tokenizer = tokenizer_from_meta(meta)
    L = int(meta["L"]) if "L" in meta else None
    vocab_size = int(meta["vocab_size"]) if "vocab_size" in meta else None

    def validate(shard: Shard, path: Path) -> None:
        try:
            shard.validate(L=L, vocab_size=vocab_size, tokenizer=tokenizer)
        except ValueError as e:
            raise ValueError(f"invalid shard {path}: {e}") from e

    return validate


class ShardDataset(Dataset):
    """Directory of ``.npz`` shards; shards are loaded lazily on first access and cached.

    Each shard is validated once when first loaded (format invariants, plus mask/tokenizer
    consistency when ``meta.json`` identifies the game).  ``__getitem__`` returns one session as
    a dict of torch tensors; ``get_batch`` gathers many rows at once (much faster than a
    ``DataLoader`` over single items for this data).
    """

    def __init__(
        self,
        data_dir: str | Path,
        indices: np.ndarray | None = None,
        max_cached: int | None = None,
        validate: bool = True,
        _store: _ShardStore | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.meta: dict[str, Any] = read_meta(self.data_dir)
        if _store is None:
            validator = _make_validator(self.meta) if validate else None
            _store = _ShardStore(shard_paths(self.data_dir), max_cached, validator)
        self._store = _store
        self.indices = (
            np.arange(self._store.total, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.indices.size and (
            self.indices.min() < 0 or self.indices.max() >= self._store.total
        ):
            raise IndexError("subset indices out of range")

    # ------------------------------------------------------------------ Dataset API
    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        g = int(self.indices[i])
        k, r = self._store.locate(np.array([g]))
        shard = self._store.shard(int(k[0]))
        row = int(r[0])
        return to_tensors({key: getattr(shard, key)[row] for key in SHARD_KEYS})

    def get_batch(self, idx: np.ndarray | Sequence[int]) -> dict[str, torch.Tensor]:
        """Gather rows ``idx`` (dataset-local indices) into one batch dict, preserving order."""
        return to_tensors(self.get_arrays(idx))

    def get_arrays(self, idx: np.ndarray | Sequence[int]) -> dict[str, np.ndarray]:
        """Like ``get_batch`` but returns numpy arrays."""
        idx = np.asarray(idx, dtype=np.int64)
        g = self.indices[idx]
        ks, rows = self._store.locate(g)
        out: dict[str, np.ndarray] = {}
        for k in np.unique(ks):
            sel = ks == k
            shard = self._store.shard(int(k))
            part = shard.take(rows[sel])
            for key in SHARD_KEYS:
                arr = getattr(part, key)
                if key not in out:
                    out[key] = np.empty((idx.size, *arr.shape[1:]), dtype=arr.dtype)
                out[key][sel] = arr
        return out

    # ------------------------------------------------------------------ views / info
    def subset(self, indices: np.ndarray | Sequence[int]) -> ShardDataset:
        """Dataset over ``self.indices[indices]`` sharing the shard cache."""
        return ShardDataset(
            self.data_dir, self.indices[np.asarray(indices, dtype=np.int64)], _store=self._store
        )

    def split(self, n_eval: int) -> tuple[ShardDataset, ShardDataset]:
        """Deterministic split: the last ``n_eval`` sessions are held out."""
        n = len(self)
        if not 0 <= n_eval < n:
            raise ValueError(f"n_eval={n_eval} must be in [0, {n})")
        return self.subset(np.arange(0, n - n_eval)), self.subset(np.arange(n - n_eval, n))

    def load_all(self) -> Shard:
        """Materialise the whole dataset (in index order) as one ``Shard`` (a full extra copy)."""
        return Shard(**self.get_arrays(np.arange(len(self))))

    @property
    def opp_ids(self) -> np.ndarray:
        """``opp_id`` of every session in this dataset, without loading full shards."""
        return self._store.all_opp_ids()[self.indices]

    @property
    def L(self) -> int:
        return int(self.meta["L"]) if "L" in self.meta else self._store.shard(0).L

    @property
    def P(self) -> int:
        return int(self.meta["theta_dim"]) if "theta_dim" in self.meta else self._store.shard(0).P

    @property
    def shard_files(self) -> list[Path]:
        return list(self._store.paths)
