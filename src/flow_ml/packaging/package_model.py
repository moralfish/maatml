from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

from ..models.manifest import ConfidenceThresholds, ModelManifest
from ..utils.io import sha256_file

console = Console()

JCL_REQUIRED = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "prompt_spec.json",
    "jcl_validation_schema.json",
    "node_contracts.json",
)
SPOOL_REQUIRED = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "prompt_spec.json",
    "spool_interpretation_schema.json",
    "node_contracts.json",
)
FLOW_GRAPH_REQUIRED = (
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "prompt_spec.json",
    "flow_graph_schema.json",
    "node_contracts.json",
)


@dataclass
class PackageResult:
    pkg_dir: Path
    manifest: ModelManifest
    files: list[str]
    fm_path: Optional[Path] = None  # set when .fm archive is also written


@dataclass
class VerifyResult:
    ok: bool
    checked_files: dict[str, bool]
    forward_ok: bool
    issues: list[str]


def _copytree_filtered(src: Path, dst: Path, *, skip_prefixes: tuple[str, ...] = ("checkpoint-",)) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.iterdir():
        if entry.is_dir() and any(entry.name.startswith(p) for p in skip_prefixes):
            continue
        target = dst / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)


def _hash_files(pkg_dir: Path, files: list[str]) -> dict[str, str]:
    return {f: sha256_file(pkg_dir / f) for f in files if (pkg_dir / f).exists()}


def _check_required(pkg_dir: Path, required: tuple[str, ...]) -> tuple[list[str], list[str]]:
    present = [f for f in required if (pkg_dir / f).exists()]
    missing = [f for f in required if not (pkg_dir / f).exists()]
    return present, missing


def _write_fm_archive(pkg_dir: Path, fm_path: Path) -> Path:
    """Pack the unpacked package directory into a single ``.fm`` archive.

    A ``.fm`` ("flow model") is a renamed zip - same on-disk layout as
    the unpacked directory, just bundled as one file the user can drag
    into Flow Studio's Models drawer. We use ``ZIP_STORED`` (no
    compression): safetensors are already binary and only round 1-3%
    smaller under DEFLATE, but the pack+unpack CPU cost added ~30-60s
    per multi-GB model on flow-studio's install path. With STORED the
    archive is effectively a tar — extraction is a plain disk copy.
    """
    fm_path.parent.mkdir(parents=True, exist_ok=True)
    if fm_path.exists():
        fm_path.unlink()
    with zipfile.ZipFile(fm_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for entry in sorted(pkg_dir.rglob("*")):
            if entry.is_file():
                arcname = entry.relative_to(pkg_dir)
                zf.write(entry, arcname.as_posix())
    return fm_path


# --- Archive validation -----------------------------------------------------
#
# A ``.fm`` is a structured archive we open from untrusted sources (Flow Studio
# users can drag arbitrary files in).  Before we extract anything we walk the
# central directory and reject:
#   - path traversal (``..`` segments, absolute paths, drive letters)
#   - symlinks (zip's external-attr 0xA000 bit)
#   - more than MAX_ENTRIES files (sane manifest is ~10 entries)
#   - cumulative declared uncompressed size over MAX_UNCOMPRESSED_BYTES
#     (zip-bomb guard; the largest legit ``.fm`` we ship is ~720 MB).

MAX_ENTRIES = 50
MAX_UNCOMPRESSED_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
_SYMLINK_MODE = 0xA000  # S_IFLNK in zip's external_attr (high bits hold mode)


def _validate_archive_entries(zf: zipfile.ZipFile) -> None:
    """Reject any zip whose entries look unsafe.  Call before extraction.

    Raises ValueError with a specific message; the caller turns this into the
    user-visible install error.
    """
    infos = zf.infolist()
    if len(infos) > MAX_ENTRIES:
        raise ValueError(
            f"archive has {len(infos)} entries (max {MAX_ENTRIES}); refusing to read"
        )
    total = 0
    for info in infos:
        name = info.filename
        # Reject anything that doesn't normalise into the archive root.
        if not name or name.startswith("/") or name.startswith("\\"):
            raise ValueError(f"archive entry has absolute path: {name!r}")
        if ":" in name.split("/", 1)[0]:
            # Windows drive letter like 'C:foo'
            raise ValueError(f"archive entry looks like a drive path: {name!r}")
        # PurePosixPath normalisation catches ".." traversal regardless of OS.
        from pathlib import PurePosixPath
        parts = PurePosixPath(name).parts
        if any(p == ".." for p in parts):
            raise ValueError(f"archive entry contains '..': {name!r}")
        # Reject symlinks: zip stores file mode in the upper 16 bits of
        # external_attr.  The 0xA000 bits indicate S_IFLNK on POSIX archivers.
        mode = (info.external_attr >> 16) & 0xF000
        if mode == _SYMLINK_MODE:
            raise ValueError(f"archive entry is a symlink: {name!r}")
        total += info.file_size
        if total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError(
                f"archive declares >{MAX_UNCOMPRESSED_BYTES} bytes uncompressed; "
                f"refusing to read"
            )


def read_manifest_from_fm(fm_path: str | Path) -> ModelManifest:
    """Read ``manifest.json`` directly from a ``.fm`` archive without extracting.

    Useful for "preview before install" UI flows: open the archive, validate
    its structure, decode just the manifest bytes, return a typed
    :class:`ModelManifest`.  No tempfile, no disk write.
    """
    fm_path = Path(fm_path)
    with zipfile.ZipFile(fm_path) as zf:
        _validate_archive_entries(zf)
        try:
            data = zf.read("manifest.json")
        except KeyError:
            raise ValueError(f"{fm_path}: archive missing manifest.json")
    return ModelManifest.model_validate_json(data)


def _build_manifest(
    *,
    model_id: str,
    task: str,
    base_checkpoint: Optional[str],
    pkg_dir: Path,
    max_input_tokens: int,
    expected_latency_ms: int,
    extra_files: list[str],
    labels_file: Optional[str] = None,
    prompt_spec_file: Optional[str] = None,
    confidence_thresholds: Optional[ConfidenceThresholds] = None,
    version: str = "v1",
    weights_dtype: str = "f32",
) -> ModelManifest:
    files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        *extra_files,
    ]
    sha = _hash_files(pkg_dir, files)
    return ModelManifest(
        model_id=model_id,
        task=task,
        max_input_tokens=max_input_tokens,
        expected_latency_ms=expected_latency_ms,
        version=version,
        base_checkpoint=base_checkpoint,
        labels_file=labels_file,
        prompt_spec_file=prompt_spec_file,
        confidence_thresholds=confidence_thresholds or ConfidenceThresholds(),
        sha256=sha,
        weights_dtype=weights_dtype,
    )


def _normalize_qwen_config(pkg_dir: Path) -> None:
    """Make the saved `config.json` candle-friendly.

    transformers 5.x nests the rope params inside ``rope_parameters``:

        "rope_parameters": { "rope_theta": 1000000, "rope_type": "default" }

    candle-transformers' Qwen2/Qwen3 ``Config`` structs still expect the
    legacy flat fields (``rope_theta`` and optional ``rope_scaling``) at
    the top level. Without this lift the runtime parse fails with
    ``missing field 'rope_theta'`` and the install rolls back. Idempotent
    — does nothing when the flat field already exists.
    """
    cfg_path = pkg_dir / "config.json"
    if not cfg_path.exists():
        return
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    rope_params = cfg.get("rope_parameters")
    if isinstance(rope_params, dict):
        if "rope_theta" not in cfg and "rope_theta" in rope_params:
            cfg["rope_theta"] = rope_params["rope_theta"]
        if (
            "rope_scaling" not in cfg
            and rope_params.get("rope_type")
            and rope_params["rope_type"] != "default"
        ):
            cfg["rope_scaling"] = {
                "type": rope_params["rope_type"],
                **{k: v for k, v in rope_params.items() if k not in ("rope_theta", "rope_type")},
            }
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")


def _convert_weights_to_dtype(pkg_dir: Path, target_dtype: str) -> None:
    """Re-save `model.safetensors` at half precision when `target_dtype` is
    not `"f32"`. Mandatory for 7B+ bases so the resulting `.fm` archive
    stays under ~16 GB (a 7B F32 dump is ~28 GB; F16 is ~14 GB; BF16 is
    the same).

    Identity short-circuit: when `target_dtype == "f32"` we leave the
    file untouched so existing JCL/spool/legacy DSL packagers stay
    bit-for-bit identical to their pre-Phase-4b output.

    Implementation note: we load via `transformers.AutoModelForCausalLM`
    rather than reading raw safetensors because the merged checkpoint
    sometimes includes optimizer / scheduler state in nearby files; the
    transformers loader is the only piece in our pipeline that knows
    which subset to keep. After the dtype cast we save back via
    `safe_serialization=True` so the runtime's mmap path stays valid.
    """
    if target_dtype == "f32":
        return
    import torch
    from transformers import AutoModelForCausalLM

    weights_path = pkg_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(
            f"_convert_weights_to_dtype: {weights_path} missing; "
            f"the packager must run after a successful train + merge step"
        )

    torch_dtype = {
        "f16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }.get(target_dtype.lower())
    if torch_dtype is None:
        raise ValueError(
            f"_convert_weights_to_dtype: unsupported target_dtype "
            f"{target_dtype!r}; expected f16 or bf16"
        )

    console.print(
        f"[cyan]converting weights to {target_dtype}[/]: {pkg_dir} ..."
    )
    model = AutoModelForCausalLM.from_pretrained(pkg_dir, torch_dtype=torch_dtype)
    # save_pretrained writes model.safetensors + config.json - exactly what
    # the runtime mmap path expects. We let it overwrite the F32 weights.
    model.save_pretrained(pkg_dir, safe_serialization=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def package_jcl(
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    *,
    prompt_spec_path: Optional[str | Path] = None,
    schema_path: Optional[str | Path] = None,
    contracts_path: Optional[str | Path] = None,
    model_id: str = "jcl-validator:v1",
    base_checkpoint: Optional[str] = "Qwen/Qwen3-1.7B",
    max_input_tokens: int = 4096,
    expected_latency_ms: int = 1500,
    confidence_thresholds: Optional[ConfidenceThresholds] = None,
    version: str = "v1",
    weights_dtype: str = "f16",
) -> PackageResult:
    """Package a generative-SFT JCL Validator checkpoint for the Candle runtime.

    Layout matches `package_flow_graph` plus the JCL-specific schema +
    contracts payload files. The runtime's 6-layer JCL validator reads
    them directly from the package.

    `weights_dtype` defaults to `"f16"` for cross-platform 16 GB-RAM
    targets — Qwen3-1.7B at fp16 packs to ~3.4 GB on disk.
    """
    src = Path(checkpoint_dir)
    dst = Path(out_dir)
    if dst.exists():
        shutil.rmtree(dst)
    _copytree_filtered(src, dst)

    spec_dst = dst / "prompt_spec.json"
    if not spec_dst.exists():
        if prompt_spec_path is None:
            raise FileNotFoundError(
                f"package_jcl: {spec_dst} missing and no --prompt-spec provided"
            )
        shutil.copy2(prompt_spec_path, spec_dst)

    schema_dst = dst / "jcl_validation_schema.json"
    if not schema_dst.exists():
        if schema_path is None:
            schema_path = (
                Path(__file__).resolve().parents[3]
                / "models"
                / "jcl-validator"
                / "datasets"
                / "jcl_validation_schema.json"
            )
        shutil.copy2(schema_path, schema_dst)

    contracts_dst = dst / "node_contracts.json"
    if not contracts_dst.exists():
        if contracts_path is None:
            contracts_path = (
                Path(__file__).resolve().parents[3]
                / "models"
                / "jcl-validator"
                / "datasets"
                / "node_contracts.json"
            )
        shutil.copy2(contracts_path, contracts_dst)

    present, missing = _check_required(dst, JCL_REQUIRED)
    if missing:
        raise FileNotFoundError(f"package_jcl: missing files in {dst}: {missing}")

    _convert_weights_to_dtype(dst, weights_dtype)
    _normalize_qwen_config(dst)

    manifest = _build_manifest(
        model_id=model_id,
        task="jcl_validation",
        base_checkpoint=base_checkpoint,
        pkg_dir=dst,
        max_input_tokens=max_input_tokens,
        expected_latency_ms=expected_latency_ms,
        extra_files=["prompt_spec.json", "jcl_validation_schema.json", "node_contracts.json"],
        prompt_spec_file="prompt_spec.json",
        confidence_thresholds=confidence_thresholds,
        version=version,
        weights_dtype=weights_dtype,
    )
    manifest.write(dst / "manifest.json")
    fm_path = _write_fm_archive(dst, dst.parent / f"{dst.name}.fm")
    console.print(
        f"[green]package jcl[/]: {dst}  (dtype={weights_dtype}, fm: {fm_path})"
    )
    return PackageResult(pkg_dir=dst, manifest=manifest, files=present, fm_path=fm_path)


def package_spool(
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    *,
    prompt_spec_path: Optional[str | Path] = None,
    schema_path: Optional[str | Path] = None,
    contracts_path: Optional[str | Path] = None,
    model_id: str = "spool-interpreter:v1",
    base_checkpoint: Optional[str] = "Qwen/Qwen3-1.7B",
    max_input_tokens: int = 4096,
    expected_latency_ms: int = 1500,
    confidence_thresholds: Optional[ConfidenceThresholds] = None,
    version: str = "v1",
    weights_dtype: str = "f16",
) -> PackageResult:
    """Package a generative-SFT Spool Interpreter checkpoint for the Candle runtime."""
    src = Path(checkpoint_dir)
    dst = Path(out_dir)
    if dst.exists():
        shutil.rmtree(dst)
    _copytree_filtered(src, dst)

    spec_dst = dst / "prompt_spec.json"
    if not spec_dst.exists():
        if prompt_spec_path is None:
            raise FileNotFoundError(
                f"package_spool: {spec_dst} missing and no --prompt-spec provided"
            )
        shutil.copy2(prompt_spec_path, spec_dst)

    schema_dst = dst / "spool_interpretation_schema.json"
    if not schema_dst.exists():
        if schema_path is None:
            schema_path = (
                Path(__file__).resolve().parents[3]
                / "models"
                / "spool-interpreter"
                / "datasets"
                / "spool_interpretation_schema.json"
            )
        shutil.copy2(schema_path, schema_dst)

    contracts_dst = dst / "node_contracts.json"
    if not contracts_dst.exists():
        if contracts_path is None:
            contracts_path = (
                Path(__file__).resolve().parents[3]
                / "models"
                / "spool-interpreter"
                / "datasets"
                / "node_contracts.json"
            )
        shutil.copy2(contracts_path, contracts_dst)

    present, missing = _check_required(dst, SPOOL_REQUIRED)
    if missing:
        raise FileNotFoundError(f"package_spool: missing files in {dst}: {missing}")

    _convert_weights_to_dtype(dst, weights_dtype)
    _normalize_qwen_config(dst)

    manifest = _build_manifest(
        model_id=model_id,
        task="spool_interpretation",
        base_checkpoint=base_checkpoint,
        pkg_dir=dst,
        max_input_tokens=max_input_tokens,
        expected_latency_ms=expected_latency_ms,
        extra_files=["prompt_spec.json", "spool_interpretation_schema.json", "node_contracts.json"],
        prompt_spec_file="prompt_spec.json",
        confidence_thresholds=confidence_thresholds,
        version=version,
        weights_dtype=weights_dtype,
    )
    manifest.write(dst / "manifest.json")
    fm_path = _write_fm_archive(dst, dst.parent / f"{dst.name}.fm")
    console.print(
        f"[green]package spool[/]: {dst}  (dtype={weights_dtype}, fm: {fm_path})"
    )
    return PackageResult(pkg_dir=dst, manifest=manifest, files=present, fm_path=fm_path)



def package_flow_graph(
    checkpoint_dir: str | Path,
    out_dir: str | Path,
    *,
    prompt_spec_path: Optional[str | Path] = None,
    schema_path: Optional[str | Path] = None,
    contracts_path: Optional[str | Path] = None,
    model_id: str = "flow-graph-generator:v1",
    base_checkpoint: Optional[str] = "Qwen/Qwen3-1.7B",
    max_input_tokens: int = 4096,
    expected_latency_ms: int = 800,
    confidence_thresholds: Optional[ConfidenceThresholds] = None,
    version: str = "v1",
    weights_dtype: str = "f16",
) -> PackageResult:
    """Package a FlowGraphGenerator checkpoint for the Candle runtime.

    Layout matches `package_dsl` plus two extra payload files:
    `flow_graph_schema.json` (the FlowGraphDto JSON Schema) and
    `node_contracts.json` (the closed vocabulary). The runtime's
    7-layer validator reads both of these directly from the package
    so the model proposal can be validated without reaching into the
    flow-studio source tree.

    `weights_dtype` defaults to `"f16"` for cross-platform 16 GB-RAM
    targets — Qwen3-1.7B at fp16 packs to ~3.4 GB on disk and ~4-5 GB
    runtime, well inside the 16 GB envelope on every supported OS.
    """
    src = Path(checkpoint_dir)
    dst = Path(out_dir)
    if dst.exists():
        shutil.rmtree(dst)
    _copytree_filtered(src, dst)

    spec_dst = dst / "prompt_spec.json"
    if not spec_dst.exists():
        if prompt_spec_path is None:
            raise FileNotFoundError(
                f"package_flow_graph: {spec_dst} missing and no --prompt-spec provided"
            )
        shutil.copy2(prompt_spec_path, spec_dst)

    schema_dst = dst / "flow_graph_schema.json"
    if not schema_dst.exists():
        if schema_path is None:
            # default to the repo's source-of-truth schema
            schema_path = (
                Path(__file__).resolve().parents[3]
                / "models"
                / "flow-graph-generator"
                / "datasets"
                / "flow_graph_schema.json"
            )
        shutil.copy2(schema_path, schema_dst)

    contracts_dst = dst / "node_contracts.json"
    if not contracts_dst.exists():
        if contracts_path is None:
            contracts_path = (
                Path(__file__).resolve().parents[3]
                / "models"
                / "flow-graph-generator"
                / "datasets"
                / "node_contracts.json"
            )
        shutil.copy2(contracts_path, contracts_dst)

    present, missing = _check_required(dst, FLOW_GRAPH_REQUIRED)
    if missing:
        raise FileNotFoundError(f"package_flow_graph: missing files in {dst}: {missing}")

    _convert_weights_to_dtype(dst, weights_dtype)
    _normalize_qwen_config(dst)

    manifest = _build_manifest(
        model_id=model_id,
        task="flow_graph_generation",
        base_checkpoint=base_checkpoint,
        pkg_dir=dst,
        max_input_tokens=max_input_tokens,
        expected_latency_ms=expected_latency_ms,
        extra_files=["prompt_spec.json", "flow_graph_schema.json", "node_contracts.json"],
        prompt_spec_file="prompt_spec.json",
        confidence_thresholds=confidence_thresholds,
        version=version,
        weights_dtype=weights_dtype,
    )
    manifest.write(dst / "manifest.json")
    fm_path = _write_fm_archive(dst, dst.parent / f"{dst.name}.fm")
    console.print(
        f"[green]package flow_graph[/]: {dst}  (dtype={weights_dtype}, fm: {fm_path})"
    )
    return PackageResult(pkg_dir=dst, manifest=manifest, files=present, fm_path=fm_path)


def verify_package(pkg_path: str | Path) -> VerifyResult:
    """Reload the package via transformers and run a one-shot forward pass.

    Accepts either an unpacked package directory or a ``.fm`` archive.

    For a ``.fm`` archive the flow is **direct read first, selective extract
    second** (see /Users/nedal/.claude/plans/...md "Phase 1"):

    1. Open the zip; run :func:`_validate_archive_entries` to catch path
       traversal, symlinks, oversize, etc. - no disk write yet.
    2. Read ``manifest.json`` directly from the archive.
    3. Extract **only** the files listed in ``manifest.sha256`` (plus the
       manifest itself) into a tempdir.  Stray entries are ignored.
    4. Run the existing sha256 + transformers forward-pass tail.
    5. Clean up the tempdir.
    """
    pkg_path = Path(pkg_path)
    tmpdir: Optional[tempfile.TemporaryDirectory] = None
    if pkg_path.is_file() and pkg_path.suffix == ".fm":
        tmpdir = tempfile.TemporaryDirectory(prefix="fm-verify-")
        try:
            with zipfile.ZipFile(pkg_path, "r") as zf:
                _validate_archive_entries(zf)
                if "manifest.json" not in zf.namelist():
                    raise ValueError(f"{pkg_path}: archive missing manifest.json")
                manifest_pre = ModelManifest.model_validate_json(zf.read("manifest.json"))
                # Extract only the manifest itself + the files listed in
                # manifest.sha256. Anything else in the archive is ignored.
                wanted = {"manifest.json", *manifest_pre.sha256.keys()}
                for name in wanted:
                    if name in zf.namelist():
                        zf.extract(name, tmpdir.name)
        except Exception as e:
            tmpdir.cleanup()
            return VerifyResult(
                ok=False, checked_files={}, forward_ok=False,
                issues=[f"archive validation failed: {e}"],
            )
        pkg = Path(tmpdir.name)
    else:
        pkg = pkg_path
    issues: list[str] = []
    manifest_path = pkg / "manifest.json"
    if not manifest_path.exists():
        if tmpdir is not None:
            tmpdir.cleanup()
        return VerifyResult(ok=False, checked_files={}, forward_ok=False, issues=["missing manifest.json"])

    manifest = ModelManifest.read(manifest_path)
    checked: dict[str, bool] = {}
    for fname, expected_hash in manifest.sha256.items():
        path = pkg / fname
        if not path.exists():
            checked[fname] = False
            issues.append(f"missing {fname}")
            continue
        actual = sha256_file(path)
        ok = actual == expected_hash
        checked[fname] = ok
        if not ok:
            issues.append(f"sha256 mismatch for {fname}")

    forward_ok = False
    try:
        if manifest.task in (
            "jcl_validation",
            "spool_interpretation",
            "flow_graph_generation",
        ):
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            tok = AutoTokenizer.from_pretrained(pkg)
            if tok.pad_token_id is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(pkg)
            model.eval()
            enc = tok("hello", return_tensors="pt")
            with torch.inference_mode():
                model(**enc)
            forward_ok = True
        else:
            issues.append(f"unknown task: {manifest.task}")
    except Exception as e:  # noqa: BLE001
        issues.append(f"forward failed: {type(e).__name__}: {e}")

    ok = all(checked.values()) and forward_ok and not issues
    if tmpdir is not None:
        tmpdir.cleanup()
    return VerifyResult(ok=ok, checked_files=checked, forward_ok=forward_ok, issues=issues)


def package_model() -> None:
    console.print(
        "Use flow_ml.packaging.package_model.package_jcl(...) / package_spool(...) "
        "/ package_flow_graph(...)"
    )
