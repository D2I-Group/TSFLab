"""Static code-similarity mining for component extraction candidates.

Nominates, never decides: the output is a ranked shortlist for the
``curate-components`` workflow, which must still prove equivalence (equations,
shapes, state, outputs, gradients) before anything is extracted or migrated.

Algorithm (pure ``ast``; no model is imported or executed):

1. Units. Every top-level class and function in a model package, every method of
   those classes, and the same for ``_components`` (components are the
   ``reuse-existing`` targets).
2. Canonical token stream. The AST is walked into node-type tokens. Names and
   attributes resolve through the module's imports: library API chains
   (``torch.fft.rfft``, ``nn.Linear``, ``F.softmax``, ``einops.rearrange``),
   builtins, and tensor methods (``.permute``, ``.softmax``) are kept verbatim;
   every other identifier or dotted operand is alpha-renamed as one name in order
   of first use (``self.proj`` -> ``self.a0``, ``config.n_embd`` -> ``v1``), so
   renamed copies and constants-versus-config variants collide. Annotations are
   not part of the stream. Exact clones hash the index-renamed stream; near-clone
   shingles use blind names (``v``, ``self.a``) so an extra argument does not
   renumber, and break, every later shingle.
   Small integers stay literal, other numbers and strings become placeholders,
   docstrings are dropped.
3. Exact clones share a hash of the canonical stream.
4. Near clones: 7-token shingles, 128-permutation MinHash (multiply-shift hashing), LSH banding (32 x 4)
   for candidate pairs, then exact shingle Jaccard. The score blends structure
   (0.7) with the Jaccard of library-call multisets (0.3), so two units that
   share layout but call different operators rank lower. Near (non-exact)
   pairs whose units call fewer than two distinct library functions are glue,
   not operators, and are dropped, as are near pairs of two ``__init__``
   methods (layer declarations alone say nothing about the computation).
5. Pairs at or above the threshold across different owners (two models, or a
   model and a component) are clustered with union-find. A cluster that
   contains a component unit suggests ``reuse-existing``; one that spans two or
   more models suggests ``extract-new``. Methods inside an already-reported class
   pair are suppressed. Clusters rank by (owners x mean score).
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SHINGLE = 7
PERMUTATIONS = 128
BANDS = 32
_ROWS = PERMUTATIONS // BANDS
_LIBRARIES = {"torch", "numpy", "np", "math", "einops", "scipy", "pywt", "F", "nn"}
_BUILTINS = set(dir(builtins))
_SMALL_INTS = {-1, 0, 1, 2}
# Renamed identifiers lose their index for near-clone shingles: one extra
# argument would otherwise renumber every later name and break all shingles.
_BLIND = re.compile(r"(?<![\w.])((?:arg:)?(?:self\.)?[av])\d+")


@dataclass
class Unit:
    owner: str  # model package slug, or "component:<name>"
    qualname: str
    path: str
    line: int
    tokens: list[str]  # blind-renamed stream for near-clone shingles
    calls: Counter
    exact_key: str = ""  # hash of the index-renamed stream (exact clones)
    parent: str | None = None  # qualname of the enclosing class for methods
    shingles: set[int] = field(default_factory=set)

    @property
    def is_component(self) -> bool:
        return self.owner.startswith("component:")

    @property
    def ref(self) -> str:
        return f"{self.owner}:{self.qualname}"


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    """Map local names to the library chain they import (``F`` -> ``torch.nn.functional``)."""
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _LIBRARIES:
                    aliases[alias.asname or root] = alias.name if alias.asname else root
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in _LIBRARIES:
                for alias in node.names:
                    aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _dotted(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


class _Canonicalizer:
    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases
        self.names: dict[str, str] = {}
        self.tokens: list[str] = []
        self.calls: Counter = Counter()

    def _rename(self, name: str, prefix: str = "v") -> str:
        key = f"{prefix}:{name}"  # attributes and locals are separate namespaces
        if key not in self.names:
            self.names[key] = f"{prefix}{len(self.names)}"
        return self.names[key]

    def _library(self, chain: list[str]) -> str | None:
        head = chain[0]
        if head in self.aliases:
            return ".".join([self.aliases[head], *chain[1:]])
        return None

    def visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.tokens.append(type(node).__name__)
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(getattr(body[0], "value", None), ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                body = body[1:]  # docstring
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    self.visit(base)
            else:
                self.visit(node.args)
            for child in body:
                self.visit(child)
            return
        if isinstance(node, (ast.Attribute, ast.Name)):
            chain = _dotted(node)
            if chain is not None:
                library = self._library(chain)
                if library is not None:
                    self.tokens.append(library)
                    return
                if chain[0] == "self":
                    self.tokens.append("self." + ".".join(self._rename(p, "a") for p in chain[1:]))
                    return
                if len(chain) == 1 and chain[0] in _BUILTINS:
                    self.tokens.append(chain[0])
                    return
                if not (len(chain) > 1 and chain[-1] in _TENSOR_METHODS):
                    # An opaque operand: `config.n_embd` and `GPT2_WIDTH` both become one name.
                    self.tokens.append(self._rename(".".join(chain)))
                    return
            if isinstance(node, ast.Attribute):
                self.visit(node.value)
                self.tokens.append("." + node.attr if node.attr in _TENSOR_METHODS else ".attr")
                return
            self.tokens.append(self._rename(node.id))
            return
        if isinstance(node, ast.arg):
            self.tokens.append("arg:" + ("self" if node.arg == "self" else self._rename(node.arg)))
            return
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, bool) or value is None:
                self.tokens.append(repr(value))
            elif isinstance(value, int) and value in _SMALL_INTS:
                self.tokens.append(str(value))
            elif isinstance(value, (int, float, complex)):
                self.tokens.append("NUM")
            else:
                self.tokens.append("STR")
            return
        if isinstance(node, ast.Call):
            chain = _dotted(node.func)
            if chain is not None:
                library = self._library(chain)
                if library is not None:
                    self.calls[library] += 1
                elif len(chain) > 1 and chain[-1] in _TENSOR_METHODS:
                    self.calls["." + chain[-1]] += 1
        if isinstance(node, (ast.operator, ast.cmpop, ast.unaryop, ast.boolop)):
            self.tokens.append(type(node).__name__)
            return
        if isinstance(node, ast.expr_context):
            return
        self.tokens.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            self.visit(child)


# Tensor methods whose names carry semantics even on alpha-renamed receivers.
_TENSOR_METHODS = {
    "permute",
    "transpose",
    "reshape",
    "view",
    "unsqueeze",
    "squeeze",
    "flatten",
    "unfold",
    "contiguous",
    "expand",
    "repeat",
    "softmax",
    "mean",
    "std",
    "var",
    "sum",
    "max",
    "min",
    "exp",
    "log",
    "sqrt",
    "abs",
    "pow",
    "matmul",
    "masked_fill",
    "detach",
    "chunk",
    "split",
    "cumsum",
    "topk",
    "gather",
    "scatter",
    "sigmoid",
    "tanh",
    "relu",
    "norm",
    "real",
    "imag",
    "conj",
    "roll",
    "flip",
    "clamp",
    "argmax",
    "argsort",
    "sort",
    "index_select",
    "narrow",
}


def _units_from_file(path: Path, owner: str, root: Path) -> list[Unit]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []
    aliases = _import_aliases(tree)
    rel = path.relative_to(root).as_posix()
    units: list[Unit] = []

    def make(node: ast.AST, qualname: str, parent: str | None) -> Unit:
        canon = _Canonicalizer(aliases)
        canon.visit(node)
        exact_key = hashlib.sha1("\x1f".join(canon.tokens).encode()).hexdigest()
        blind = [_BLIND.sub(r"\1", token) for token in canon.tokens]
        return Unit(owner, qualname, rel, node.lineno, blind, canon.calls, exact_key, parent)

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            units.append(make(node, node.name, None))
        elif isinstance(node, ast.ClassDef):
            units.append(make(node, node.name, None))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    units.append(make(item, f"{node.name}.{item.name}", node.name))
    return units


def collect_units(root: Path, models: set[str] | None = None) -> list[Unit]:
    """Collect model-local and component units under ``src/tsflab/models``."""
    base = root / "src" / "tsflab" / "models"
    units: list[Unit] = []
    for package in sorted(p for p in base.iterdir() if p.is_dir()):
        if package.name == "_components":
            for component in sorted(p for p in package.iterdir() if p.is_dir() and not p.name.startswith("_")):
                for path in sorted(component.rglob("*.py")):
                    units.extend(_units_from_file(path, f"component:{component.name}", root))
            continue
        if package.name.startswith("_") or (models is not None and package.name not in models):
            continue
        for path in sorted(package.rglob("*.py")):
            if path.name in {"spec.py", "__init__.py"}:
                continue
            units.extend(_units_from_file(path, package.name, root))
    return units


def _shingles(tokens: list[str]) -> set[int]:
    if len(tokens) < SHINGLE:
        return set()
    return {
        int.from_bytes(hashlib.blake2b("\x1f".join(tokens[i : i + SHINGLE]).encode(), digest_size=8).digest(), "little")
        for i in range(len(tokens) - SHINGLE + 1)
    }


def _minhash(shingles: set[int], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """MinHash by multiply-shift universal hashing (uint64 wrap-around, upper 32 bits)."""
    values = np.fromiter(shingles, dtype=np.uint64, count=len(shingles))
    with np.errstate(over="ignore"):
        hashed = (a[:, None] * values[None, :] + b[:, None]) >> np.uint64(32)
    return hashed.min(axis=1)


def _jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _counter_jaccard(left: Counter, right: Counter) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    return sum(min(left[k], right[k]) for k in keys) / sum(max(left[k], right[k]) for k in keys)


@dataclass
class Pair:
    left: Unit
    right: Unit
    structure: float
    calls: float
    exact: bool

    @property
    def score(self) -> float:
        return 1.0 if self.exact else 0.7 * self.structure + 0.3 * self.calls


def similar_pairs(units: list[Unit], *, threshold: float = 0.6, min_tokens: int = 40) -> list[Pair]:
    """Return cross-owner unit pairs scoring at least ``threshold``."""
    pool = [u for u in units if len(u.tokens) >= min_tokens]
    for unit in pool:
        unit.shingles = _shingles(unit.tokens)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 1 << 63, size=PERMUTATIONS, dtype=np.uint64) * np.uint64(2) + np.uint64(1)
    b = rng.integers(0, 1 << 63, size=PERMUTATIONS, dtype=np.uint64)
    buckets: dict[tuple, list[int]] = defaultdict(list)
    exact: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(pool):
        exact[unit.exact_key].append(index)
        signature = _minhash(unit.shingles, a, b)
        for band in range(BANDS):
            key = (band, signature[band * _ROWS : (band + 1) * _ROWS].tobytes())
            buckets[key].append(index)
    candidates: set[tuple[int, int]] = set()
    for members in list(buckets.values()) + list(exact.values()):
        if 1 < len(members) <= 200:
            for i, left in enumerate(members):
                for right in members[i + 1 :]:
                    candidates.add((min(left, right), max(left, right)))
    exact_ids = {frozenset(m) for m in exact.values() if len(m) > 1}
    pairs: list[Pair] = []
    for i, j in sorted(candidates):
        left, right = pool[i], pool[j]
        if left.owner == right.owner:
            continue
        is_exact = any(i in group and j in group for group in exact_ids)
        library = {name for name in left.calls | right.calls if not name.startswith(".")}
        if not is_exact and len(library) < 2:
            continue  # glue (reshapes and calls into sub-modules) is not an operator
        if not is_exact and left.qualname.endswith(".__init__") and right.qualname.endswith(".__init__"):
            continue  # similar layer declarations alone say nothing about the computation
        pair = Pair(
            left, right, _jaccard(left.shingles, right.shingles), _counter_jaccard(left.calls, right.calls), is_exact
        )
        if pair.score >= threshold:
            pairs.append(pair)
    # A method pair inside an already-reported class pair adds no information.
    class_pairs = {
        frozenset({(p.left.owner, p.left.qualname), (p.right.owner, p.right.qualname)})
        for p in pairs
        if p.left.parent is None and p.right.parent is None
    }
    return [
        p
        for p in pairs
        if not (
            p.left.parent
            and p.right.parent
            and frozenset({(p.left.owner, p.left.parent), (p.right.owner, p.right.parent)}) in class_pairs
        )
    ]


def clusters(pairs: list[Pair]) -> list[dict]:
    """Union-find clusters with a suggested decision, ranked for review."""
    parent: dict[str, str] = {}
    units: dict[str, Unit] = {}

    def find(key: str) -> str:
        while parent.setdefault(key, key) != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for pair in pairs:
        units[pair.left.ref] = pair.left
        units[pair.right.ref] = pair.right
        parent[find(pair.left.ref)] = find(pair.right.ref)
    groups: dict[str, list[str]] = defaultdict(list)
    for ref in units:
        groups[find(ref)].append(ref)
    scores: dict[str, list[float]] = defaultdict(list)
    for pair in pairs:
        scores[find(pair.left.ref)].append(pair.score)
    out = []
    for root_ref, refs in groups.items():
        members = sorted((units[r] for r in refs), key=lambda u: (u.is_component, u.owner, u.qualname))
        components = sorted({u.owner.split(":", 1)[1] for u in members if u.is_component})
        models = sorted({u.owner for u in members if not u.is_component})
        calls = Counter()
        for unit in members:
            calls.update(unit.calls)
        if components:
            decision = f"reuse-existing candidate: {', '.join(components)}"
        elif len(models) >= 2:
            decision = "extract-new candidate"
        else:
            decision = "review"
        out.append(
            {
                "decision": decision,
                "models": models,
                "components": components,
                "max_score": round(max(scores[root_ref]), 3),
                "mean_score": round(sum(scores[root_ref]) / len(scores[root_ref]), 3),
                "library_calls": [name for name, _ in calls.most_common(8)],
                "members": [
                    {"owner": u.owner, "unit": u.qualname, "location": f"{u.path}:{u.line}", "tokens": len(u.tokens)}
                    for u in members
                ],
            }
        )
    # Breadth times agreement: a cluster spanning more models ranks higher, but
    # only as far as its members actually agree.
    out.sort(key=lambda c: (-(len(c["models"]) + len(c["components"])) * c["mean_score"], -c["max_score"]))
    return out


def reuse_candidates(
    root: Path, slug: str, *, threshold: float = 0.6, min_tokens: int = 40, top: int = 5
) -> list[Pair]:
    """Units of model ``slug`` that resemble a shared component, most similar first.

    ``tsf model add`` reports them so a new model reuses existing units instead of
    re-implementing them; a nomination only, equivalence is proved before switching.
    """
    pairs = similar_pairs(collect_units(root, {slug}), threshold=threshold, min_tokens=min_tokens)
    hits = [pair for pair in pairs if pair.left.is_component != pair.right.is_component]
    hits.sort(key=lambda pair: -pair.score)
    return hits[:top]


def similarity_command(argv: list[str], root: Path) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="tsf model similar",
        description="Rank static code-similarity clusters as component extraction candidates "
        "(nominations only; curate-components proves equivalence).",
    )
    parser.add_argument("models", nargs="*", help="limit model packages (slugs); default: all")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--min-tokens", type=int, default=40)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    units = collect_units(root, set(args.models) if args.models else None)
    ranked = clusters(similar_pairs(units, threshold=args.threshold, min_tokens=args.min_tokens))
    shown = ranked[: args.top] if args.top else ranked
    if args.json:
        print(json.dumps({"units": len(units), "clusters": len(ranked), "shown": shown}, indent=2))
        return 0
    print(f"{len(units)} units scanned; {len(ranked)} clusters at score >= {args.threshold}")
    for index, cluster in enumerate(shown, 1):
        print(
            f"\n[{index}] {cluster['decision']}  models={len(cluster['models'])}  "
            f"max={cluster['max_score']} mean={cluster['mean_score']}"
        )
        if cluster["library_calls"]:
            print("    calls: " + ", ".join(cluster["library_calls"]))
        for member in cluster["members"]:
            print(f"    {member['owner']}:{member['unit']}  {member['location']}  ({member['tokens']} tok)")
    return 0
