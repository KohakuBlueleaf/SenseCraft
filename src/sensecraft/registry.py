"""The term registry: every loss under one name, with the input range and layout it takes."""

import difflib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum


class Range(Enum):
    """The value range a term reads its images in."""

    UNIT = "unit"  # [0, 1]
    SYMMETRIC = "symmetric"  # [-1, 1]
    ANY = "any"  # as given


@dataclass(frozen=True)
class TermInfo:
    """How a registered term is built and fed."""

    factory: Callable
    range: Range
    dims: str  # "2d": images, run per frame on video; "3d": video only; "any": either
    fp32: bool  # runs with autocast off on fp32 inputs


TERMS: dict[str, TermInfo] = {}
DIMS = ("2d", "3d", "any")


def register(
    name: str, range: Range = Range.ANY, dims: str = "any", fp32: bool = False
) -> Callable:
    """Class decorator: add a term under `name`."""
    if dims not in DIMS:
        raise ValueError(f"dims {dims!r} not in {DIMS}")

    def add(factory: Callable) -> Callable:
        if name in TERMS:
            raise ValueError(f"term {name!r} is registered twice")
        TERMS[name] = TermInfo(factory, range, dims, fp32)
        return factory

    return add


def lookup(name: str) -> TermInfo:
    """The term registered as `name`; an unknown name lists the closest ones."""
    if name in TERMS:
        return TERMS[name]
    close = difflib.get_close_matches(name, TERMS, n=3)
    hint = f"; did you mean {', '.join(close)}?" if close else ""
    raise KeyError(f"unknown term {name!r}{hint}")
