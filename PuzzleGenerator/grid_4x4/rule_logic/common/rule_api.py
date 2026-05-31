from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


Candidate = Any
Domains = list[int]


@runtime_checkable
class RuleSpecProtocol(Protocol):
    name: str

    def generate_candidates(
        self,
        solution_grid: list[int],
        *,
        rng,
        max_candidates: int | None = None,
    ) -> list[Candidate]: ...

    def solution_satisfies(
        self,
        solution_grid: list[int],
        candidate: Candidate,
    ) -> bool: ...

    def propagate(
        self,
        domains: Domains,
        candidate: Candidate,
        *,
        ctx: Any | None = None,
    ) -> bool: ...

    def candidate_key(self, candidate: Candidate) -> tuple: ...

    def describe(self, candidate: Candidate) -> str: ...

    def to_jsonable(self, candidate: Candidate) -> Any: ...


@dataclass(frozen=True)
class ActiveRule:
    spec: RuleSpecProtocol
    candidate: Candidate

    @property
    def name(self) -> str:
        return self.spec.name


@dataclass(frozen=True)
class RuleSpec:
    name: str
    generate_candidates_fn: Callable[..., list[Candidate]]
    solution_satisfies_fn: Callable[[list[int], Candidate], bool]
    propagate_fn: Callable[..., bool]
    candidate_key_fn: Callable[[Candidate], tuple]
    describe_fn: Callable[[Candidate], str]
    to_jsonable_fn: Callable[[Candidate], Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def generate_candidates(self, solution_grid: list[int], *, rng, max_candidates: int | None = None) -> list[Candidate]:
        return self.generate_candidates_fn(solution_grid, rng=rng, max_candidates=max_candidates)

    def solution_satisfies(self, solution_grid: list[int], candidate: Candidate) -> bool:
        return self.solution_satisfies_fn(solution_grid, candidate)

    def propagate(self, domains: Domains, candidate: Candidate, *, ctx: Any | None = None) -> bool:
        return self.propagate_fn(domains, candidate, ctx=ctx)

    def candidate_key(self, candidate: Candidate) -> tuple:
        return self.candidate_key_fn(candidate)

    def describe(self, candidate: Candidate) -> str:
        return self.describe_fn(candidate)

    def to_jsonable(self, candidate: Candidate) -> Any:
        return self.to_jsonable_fn(candidate)


def make_passthrough_rule(
    *,
    name: str,
    generate_candidates_fn: Callable[..., list[Candidate]],
    solution_satisfies_fn: Callable[[list[int], Candidate], bool],
    propagate_fn: Callable[..., bool],
    candidate_key_fn: Callable[[Candidate], tuple],
    describe_fn: Callable[[Candidate], str],
    to_jsonable_fn: Callable[[Candidate], Any],
    metadata: dict[str, Any] | None = None,
) -> RuleSpec:
    return RuleSpec(
        name=name,
        generate_candidates_fn=generate_candidates_fn,
        solution_satisfies_fn=solution_satisfies_fn,
        propagate_fn=propagate_fn,
        candidate_key_fn=candidate_key_fn,
        describe_fn=describe_fn,
        to_jsonable_fn=to_jsonable_fn,
        metadata={} if metadata is None else dict(metadata),
    )