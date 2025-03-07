## PL_ifelse

This is an answer to the question, why can't when/then only do the operation for the rows that are true.

Now you can!

## Usage

This tries to keep the syntax as similar to when/then as possible while having it behave like an if/elseif block. Here's an example

```python
import polars as pl
from polars import col as c
from pl_ifelse import pl_if

df = pl.DataFrame(
    {"a": [x * 1.1 for x in range(10)], "b": [x * 2.1 for x in range(10)]}
)

df.with_columns(
    z=(
        pl_if(c.a > 8)
        .then(c.b)
        .else_if(c.a > 5)
        .then(c.b + 100)
        .otherwise(c.b*10, dtype=pl.Float64)
    )
)
shape: (10, 3)
┌─────┬──────┬───────┐
│ a   ┆ b    ┆ z     │
│ --- ┆ ---  ┆ ---   │
│ f64 ┆ f64  ┆ f64   │
╞═════╪══════╪═══════╡
│ 0.0 ┆ 0.0  ┆ 0.0   │
│ 1.1 ┆ 2.1  ┆ 21.0  │
│ 2.2 ┆ 4.2  ┆ 42.0  │
│ 3.3 ┆ 6.3  ┆ 63.0  │
│ 4.4 ┆ 8.4  ┆ 84.0  │
│ 5.5 ┆ 10.5 ┆ 110.5 │
│ 6.6 ┆ 12.6 ┆ 112.6 │
│ 7.7 ┆ 14.7 ┆ 114.7 │
│ 8.8 ┆ 16.8 ┆ 16.8  │
│ 9.9 ┆ 18.9 ┆ 18.9  │
└─────┴──────┴───────┘
```

There are two syntax differences are:
1. It always has to end in an `otherwise`. Even ifyour if/then is logically complete the `otherwise` method is where it converts the chain into a polars Expr.
2. You have to tell it what the dtype is of the output.

A usability limitation: The plugin is doing the computation so the operations won't be in a lazy diagram. It's also set to be an elementwise return so if you're expecting aggregates, it might be weird results. I haven't tested it fully.

## How it works

On the python side there's the `pl_if` function which returns the `IF` class. The `IF` class returns the `THEN` class. That class has two methods, `else_if` and `otherwise`. `else_if` returns the `IF` class whereas `otherwise` launches the plugin. At each part of the chain, it extends a list of those expressions. 

When `otherwise` is called, it needs to return `register_plugin_function` with a list of `args` which will be the column data and a serializable set of kwargs. Unfortunately, the args can't just be `pl.all` so it needs to inspect all the expressions to generate a concrete list of columns. To get the expressions to the plugin it needs to serialize them into json. It turns the `dtype` that you enter into a `pl.lit(None, dtype)` which is inserted as the first member of `args`. The plugin can then tell the main polars thread that its output will be the same dtype as the first column polars wants to hand over to it.

Once the plugin gets the data, it reconstructs a df adding a row index column, partitions it by the first condition, does the first operation on the true partition and puts those results in a Vec. It then repeats that process for each subsequent if/then starting with the previous iteration's false partition until there are either no more rows or no more conditions. When all the results are in, it sorts it by the row index and returns the Series to the main polars thread.