from .ir import IRNode, IRGraph


def __getattr__(name):
    if name == "Compiler":
        from .compiler import Compiler
        return Compiler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["Compiler", "IRNode", "IRGraph"]
