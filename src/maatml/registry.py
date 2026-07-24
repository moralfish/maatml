"""Decorator-based plugin registries for trainers, validators, metrics, etc.

Plugins can come from:
  - Decorators in in-tree modules (``@register_trainer(...)``)
  - Setuptools entry points group ``maatml.plugins``
  - Per-model ``model.yml`` ``plugins:`` list (module paths or folder-local packages)
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


@dataclass(frozen=True)
class PluginEntry:
    """A registered callable plus where it came from (for ``maatml plugins``)."""

    name: str
    obj: Any
    source: str  # e.g. "decorator:maatml.training.sft_base" or "entry_point:..."


class _Registry:
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._entries: dict[str, PluginEntry] = {}

    def register(
        self, name: str, obj: Any, *, source: str = "unknown"
    ) -> Any:
        # Allow overwrite so discover_plugins(force=True) / module reload can
        # re-bind the same plugin name after a registry wipe.
        self._entries[name] = PluginEntry(name=name, obj=obj, source=source)
        return obj

    def get(self, name: str, default: Any = None) -> Any:
        entry = self._entries.get(name)
        return entry.obj if entry is not None else default

    def require(self, name: str) -> Any:
        entry = self._entries.get(name)
        if entry is None:
            known = ", ".join(sorted(self._entries)) or "(none)"
            raise KeyError(f"Unknown {self.kind} plugin {name!r}. Known: {known}")
        return entry.obj

    def names(self) -> list[str]:
        return sorted(self._entries)

    def items(self) -> list[PluginEntry]:
        return [self._entries[k] for k in sorted(self._entries)]

    def unregister(self, name: str) -> bool:
        """Drop one registration. Returns True when something was removed."""
        return self._entries.pop(name, None) is not None

    def snapshot(self) -> dict[str, PluginEntry]:
        """Copy of the current registrations (restore with :meth:`restore`)."""
        return dict(self._entries)

    def restore(self, entries: dict[str, PluginEntry]) -> None:
        """Replace registrations with a previous :meth:`snapshot`."""
        self._entries.clear()
        self._entries.update(entries)

    def clear(self) -> None:
        """Wipe registrations (tests, and ``discover_plugins(force=True)``)."""
        self._entries.clear()


TRAINERS = _Registry("trainer")
VALIDATORS = _Registry("validator")
METRICS = _Registry("metrics")
PREDICTORS = _Registry("predictor")
FORMATS = _Registry("format")
SCAFFOLD_HOOKS = _Registry("scaffold_hook")
SANITIZERS = _Registry("sanitizer")
TRANSFORMS = _Registry("transform")
EXPORTERS = _Registry("exporter")
GENERATORS = _Registry("generator")

_ALL_REGISTRIES: dict[str, _Registry] = {
    "trainer": TRAINERS,
    "validator": VALIDATORS,
    "metrics": METRICS,
    "predictor": PREDICTORS,
    "format": FORMATS,
    "scaffold_hook": SCAFFOLD_HOOKS,
    "sanitizer": SANITIZERS,
    "transform": TRANSFORMS,
    "exporter": EXPORTERS,
    "generator": GENERATORS,
}

_discovered = False


def _caller_source(extra: str = "") -> str:
    """Best-effort module name of the registering caller."""
    import inspect

    frame = inspect.currentframe()
    try:
        # decorator helper -> register_* -> caller module
        caller = frame.f_back.f_back if frame and frame.f_back else None  # type: ignore[union-attr]
        mod = caller.f_globals.get("__name__", "unknown") if caller else "unknown"
    finally:
        del frame
    return f"decorator:{mod}" + (f":{extra}" if extra else "")


def _make_decorator(registry: _Registry) -> Callable[[str], Callable[[F], F]]:
    def decorator(name: str) -> Callable[[F], F]:
        def wrapper(fn: F) -> F:
            registry.register(name, fn, source=_caller_source(name))
            return fn

        return wrapper

    return decorator


register_trainer = _make_decorator(TRAINERS)
register_validator = _make_decorator(VALIDATORS)
register_metrics = _make_decorator(METRICS)
register_predictor = _make_decorator(PREDICTORS)
register_format = _make_decorator(FORMATS)
register_scaffold_hook = _make_decorator(SCAFFOLD_HOOKS)
register_sanitizer = _make_decorator(SANITIZERS)
register_transform = _make_decorator(TRANSFORMS)
register_exporter = _make_decorator(EXPORTERS)
register_generator = _make_decorator(GENERATORS)


def get_registry(kind: str) -> _Registry:
    try:
        return _ALL_REGISTRIES[kind]
    except KeyError as exc:
        raise KeyError(
            f"Unknown registry kind {kind!r}. Known: {', '.join(_ALL_REGISTRIES)}"
        ) from exc


def list_all_plugins() -> dict[str, list[PluginEntry]]:
    """Return every registered plugin grouped by registry kind."""
    return {kind: reg.items() for kind, reg in _ALL_REGISTRIES.items()}


def snapshot_registries() -> dict[str, dict[str, PluginEntry]]:
    """Snapshot every registry (public API for tests and embedding hosts)."""
    return {kind: reg.snapshot() for kind, reg in _ALL_REGISTRIES.items()}


def restore_registries(snapshot: dict[str, dict[str, PluginEntry]]) -> None:
    """Restore a :func:`snapshot_registries` result."""
    for kind, entries in snapshot.items():
        _ALL_REGISTRIES[kind].restore(entries)


def reset_registries(*, rediscover: bool = False) -> None:
    """Wipe every registry. With ``rediscover``, re-import the built-ins after."""
    global _discovered
    for reg in _ALL_REGISTRIES.values():
        reg.clear()
    _discovered = False
    if rediscover:
        discover_plugins(force=True)


def _load_module_from_path(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_package_from_dir(pkg_dir: Path, module_name: str) -> Any:
    """Load a directory containing ``__init__.py`` as a package.

    Registers the package in ``sys.modules`` with
    ``submodule_search_locations`` so intra-package imports
    (``from .validator import ...``) resolve correctly.
    """
    init_py = pkg_dir / "__init__.py"
    if not init_py.is_file():
        raise ImportError(f"Plugin package missing __init__.py: {pkg_dir}")
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_py,
        submodule_search_locations=[str(pkg_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load plugin package from {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_model_plugins(
    model_dir: str | Path, plugin_list: Optional[Iterable[str]] = None
) -> list[str]:
    """Load plugins declared in a model's ``plugins:`` list.

    Each entry is either:
      - a dotted module path (``my_pkg.plugins.foo``),
      - a model-folder-relative ``.py`` file (``plugins/local_hook.py``), or
      - a model-folder-relative package directory (``./jcl_plugin``) containing
        ``__init__.py``.
    Returns the list of loaded module names.
    """
    model_dir = Path(model_dir).resolve()
    loaded: list[str] = []
    for entry in plugin_list or []:
        entry = entry.strip()
        if not entry:
            continue
        if entry.endswith(".py") or "/" in entry or entry.startswith("."):
            path = Path(entry)
            if not path.is_absolute():
                path = (model_dir / entry).resolve()
            if path.is_dir():
                mod_name = f"maatml._model_plugins.{model_dir.name}.{path.name}"
                _load_package_from_dir(path, mod_name)
                loaded.append(mod_name)
            elif path.is_file():
                mod_name = f"maatml._model_plugins.{model_dir.name}.{path.stem}"
                _load_module_from_path(path, mod_name)
                loaded.append(mod_name)
            else:
                raise FileNotFoundError(
                    f"Model plugin not found: {path} (from plugins: {entry!r})"
                )
        else:
            importlib.import_module(entry)
            loaded.append(entry)
    return loaded


def discover_plugins(*, force: bool = False) -> None:
    """Import built-in modules + entry-point plugins.

    Idempotent unless ``force=True`` or the trainer registry is empty (e.g.
    after a test wipe). Model-folder plugins are loaded separately via
    ``load_model_plugins``.
    """
    global _discovered
    trainers_empty = not TRAINERS.names()
    if _discovered and not force and not trainers_empty:
        return

    if force or trainers_empty:
        for reg in _ALL_REGISTRIES.values():
            reg.clear()

    modules = (
        "maatml.training.builtins",
        "maatml.data.pipeline",
        "maatml.data.formats",
        "maatml.data.preference",
        "maatml.evaluation.predictors",
        "maatml.export.bundle",
        "maatml.export.gguf",
        "maatml.export.mlx_export",
    )
    for mod in modules:
        try:
            if mod in sys.modules and (force or trainers_empty):
                importlib.reload(sys.modules[mod])
            else:
                importlib.import_module(mod)
        except ImportError:
            pass

    # Setuptools / importlib entry points.
    try:
        from importlib.metadata import entry_points

        eps = entry_points()
        selected: Any
        if hasattr(eps, "select"):
            selected = eps.select(group="maatml.plugins")
        else:
            selected = eps["maatml.plugins"]  # type: ignore[index]
        for ep in selected:
            loaded = ep.load()
            if callable(loaded) and not isinstance(loaded, type):
                try:
                    loaded()
                except TypeError:
                    pass
    except Exception:  # noqa: BLE001  discovery must not abort CLI startup
        pass

    _discovered = True
