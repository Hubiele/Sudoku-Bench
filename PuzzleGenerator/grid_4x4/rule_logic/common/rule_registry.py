from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any

try:
    from TEST.grid_4x4.rule_logic.common.rule_api import RuleSpecProtocol
except ImportError:
    from .rule_api import RuleSpecProtocol


DEFAULT_RULE_PACKAGE = "TEST.grid_4x4.rule_logic"
DEFAULT_PREFIX = "_new_arch_"
DEFAULT_SUFFIX = "_rule"


@dataclass(frozen=True)
class RegisteredRule:
    module_name: str
    module: Any
    rule_name: str
    rule_spec: RuleSpecProtocol


def _iter_rule_module_names(
    package_name: str = DEFAULT_RULE_PACKAGE,
    *,
    prefix: str = DEFAULT_PREFIX,
    suffix: str = DEFAULT_SUFFIX,
) -> list[str]:
    pkg = importlib.import_module(package_name)

    if not hasattr(pkg, "__path__"):
        raise ValueError(f"{package_name!r} er ikke en pakke")

    names: list[str] = []
    for _, module_name, ispkg in pkgutil.iter_modules(pkg.__path__):
        if ispkg:
            continue
        if not module_name.startswith(prefix):
            continue
        if suffix and not module_name.endswith(suffix):
            continue
        names.append(module_name)
    return sorted(names)


def discover_rule_modules(
    package_name: str = DEFAULT_RULE_PACKAGE,
    *,
    prefix: str = DEFAULT_PREFIX,
    suffix: str = DEFAULT_SUFFIX,
) -> list[Any]:
    modules = []
    for module_name in _iter_rule_module_names(package_name, prefix=prefix, suffix=suffix):
        full_name = f"{package_name}.{module_name}"
        modules.append(importlib.import_module(full_name))
    return modules


def validate_rule_module(module: Any) -> RegisteredRule:
    if not hasattr(module, "RULE_NAME"):
        raise ValueError(f"{module.__name__}: mangler RULE_NAME")
    if not hasattr(module, "RULE_SPEC"):
        raise ValueError(f"{module.__name__}: mangler RULE_SPEC")

    rule_name = getattr(module, "RULE_NAME")
    rule_spec = getattr(module, "RULE_SPEC")

    if getattr(rule_spec, "name", None) != rule_name:
        raise ValueError(
            f"{module.__name__}: RULE_SPEC.name ({getattr(rule_spec, 'name', None)!r}) "
            f"matcher ikke RULE_NAME ({rule_name!r})"
        )

    required = (
        "generate_candidates",
        "solution_satisfies",
        "propagate",
        "candidate_key",
        "describe",
        "to_jsonable",
    )
    for attr in required:
        if not hasattr(rule_spec, attr):
            raise ValueError(f"{module.__name__}: RULE_SPEC mangler {attr}")

    return RegisteredRule(
        module_name=module.__name__,
        module=module,
        rule_name=rule_name,
        rule_spec=rule_spec,
    )


def load_registered_rules(
    package_name: str = DEFAULT_RULE_PACKAGE,
    *,
    prefix: str = DEFAULT_PREFIX,
    suffix: str = DEFAULT_SUFFIX,
) -> dict[str, RegisteredRule]:
    out: dict[str, RegisteredRule] = {}
    for module in discover_rule_modules(package_name, prefix=prefix, suffix=suffix):
        registered = validate_rule_module(module)
        if registered.rule_name in out:
            raise ValueError(f"Dobbel regelnavn oppdaget: {registered.rule_name!r}")
        out[registered.rule_name] = registered
    return out