from __future__ import annotations

from pathlib import Path

import polars as pl
from polars import Expr
from polars.plugins import register_plugin_function


def pl_if(condition: Expr):
    return IF([condition])


class IF:
    def __init__(self, previous: list[Expr]):
        self.previous = previous

    def then(self, expr: Expr):
        return THEN(self.previous + [expr])


class THEN:
    def __init__(self, previous: list[Expr]):
        self.previous = previous

    def else_if(self, expr: Expr):
        return IF(self.previous + [expr])

    def otherwise(self, expr: Expr, *, dtype: pl.datatypes.DataTypeClass) -> Expr:
        exprs = self.previous + [expr]
        pyexprs = [x.meta.serialize(format="json") for x in exprs]
        cols = []
        while len(exprs) > 0:
            expr = exprs.pop(0)
            if expr.meta.is_column():
                if all(not expr.meta.eq(x) for x in cols):
                    cols.append(expr)
            else:
                new_exprs = expr.meta.pop()
                exprs.extend(new_exprs)

        args = [pl.lit(None, dtype=dtype), *cols]
        return register_plugin_function(
            plugin_path=Path(__file__).parent,
            function_name="if_else",
            args=args,
            kwargs={"exprs": pyexprs},
            is_elementwise=True,
        )
