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

## How it performs....it depends


Here are a few examples to compare, first an unbalanced one.

```python
import polars as pl
from polars import col as c
from pl_ifelse import pl_if
import numpy as np
import time

#Setup
df = (
    pl.DataFrame(
        {
            "a": np.random.choice([x for x in range(5)], 50_000_000),
            "b": np.random.standard_normal(50_000_000),
            "c": np.random.standard_normal(50_000_000),
        }
    )
)

## Test 1
## Base Case
strt=time.time()
df.select((pl.arctan2("a","b")**2.45).sin())
print(time.time()-strt)
# 5.08

## pl_if
strt=time.time()
df.select(pl_if(c.a==0).then((pl.arctan2("b","c")**2.45).sin()).otherwise(c.b, dtype=pl.Float64))
print(time.time()-strt)
# 3.90 seconds

## when/then
strt=time.time()
df.select(pl.when(c.a==0).then((pl.arctan2("b","c")**2.45).sin()).otherwise(c.b))
print(time.time()-strt)
# 6.28 seconds
```

So in this example doing arctan2 raised to the power of 2.45 and then taking the sin of that takes about 5.08 seconds. The problem is contrived so that `pl_if` only does the hard problem on about a fifth of the data yet it still takes ~77% of the time. Not too surpringly, if I bump up the conditional to `c.a<=2` so that it's doing slightly more than half of the data, it takes 5.08 seconds. Doing all the math sequentially instead of in parallel as when/then is going to take a very niche case to outperform when/then.


A less contrived example, let's say you do the conditional for all 5 groups of `a` with just multiplication for each operation

```python
strt=time.time()
df.with_columns(
    z=pl_if(c.a==0)
    .then(c.b*2)
    .else_if(c.a==1)
    .then(c.b*3)
    .else_if(c.a==2)
    .then(c.b*4)
    .else_if(c.a==3)
    .then(c.b*5)
    .otherwise(c.b*6, dtype=pl.Float64)
)
print(time.time()-strt)
# 4.77

strt=time.time()
df.with_columns(
    z=pl.when(c.a==0)
    .then(c.b*2)
    .when(c.a==1)
    .then(c.b*3)
    .when(c.a==2)
    .then(c.b*4)
    .when(c.a==3)
    .then(c.b*5)
    .otherwise(c.b*6)
)
print(time.time()-strt)
# 1.99
```
Even though the second example here is doing 5 operations on 50m rows for 250m operations it's still drastically outperforming the first one which is only doing 50m operations. The reasoning is that splitting up the data and then piecing it back together is more expensive than doing multiplication 200m times.


Another example using a hard problem for all chunks
```python
strt = time.time()
df.with_columns(
    z=pl_if(c.a == 0)
    .then((pl.arctan2("a","b")**2.25).sin())
    .else_if(c.a == 1)
    .then((pl.arctan2("a","b")**2.15).sin())
    .else_if(c.a == 2)
    .then((pl.arctan2("a","b")**3.45).sin())
    .else_if(c.a == 3)
    .then((pl.arctan2("a","b")**1.45).sin())
    .otherwise((pl.arctan2("a","b")**1.65).sin(), dtype=pl.Float64)
)
print(time.time() - strt)
# 6.64


strt = time.time()
df.with_columns(
    z=pl.when(c.a == 0)
    .then((pl.arctan2("a","b")**2.25).sin())
    .when(c.a == 1)
    .then((pl.arctan2("a","b")**2.15).sin())
    .when(c.a == 2)
    .then((pl.arctan2("a","b")**3.45).sin())
    .when(c.a == 3)
    .then((pl.arctan2("a","b")**1.45).sin())
    .otherwise((pl.arctan2("a","b")**1.65).sin())
)
print(time.time() - strt)
# 19.25
```

This one is a bit more surprising, it didn't seem I was getting much of any parallelization in either case. I would have expected to see 5 cores maxed out but it only seemed one was crunching. 

If I convert the last one to

```python
strt = time.time()
df.with_columns(
q=(pl.arctan2("a","b")**2.25).sin(),
w=(pl.arctan2("a","b")**2.15).sin(),
e=(pl.arctan2("a","b")**3.45).sin(),
r=(pl.arctan2("a","b")**1.45).sin(),
t=(pl.arctan2("a","b")**1.65).sin()
).select('a','b',
    z=pl.when(c.a == 0)
    .then("q")
    .when(c.a == 1)
    .then("w")
    .when(c.a == 2)
    .then("e")
    .when(c.a == 3)
    .then('r')
    .otherwise('t')
)
print(time.time() - strt)
#6.84
```

Now it's back to being on par with the sequential one.