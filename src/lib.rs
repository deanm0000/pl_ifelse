use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use pyo3_polars::PolarsAllocator;
use serde::Deserialize;
#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule(name = "_pl_ifelse")]
fn _pl_ifelse(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    Ok(())
}

pub fn first_output(fields: &[Field]) -> PolarsResult<Field> {
    Ok(fields[0].clone())
}
#[derive(Deserialize)]
struct InKwargs {
    exprs: Vec<String>,
}

#[polars_expr(output_type_func=first_output)]
fn if_else(inputs: &[Series], kwargs: InKwargs) -> PolarsResult<Series> {

    let cols: Vec<Column> = inputs[1..inputs.len()]
        .iter()
        .map(|s| s.clone().into_column())
        .collect();
    let mut df = DataFrame::new(cols)?;
    df = df.with_row_index("___row_index".into(), None)?;
    let mut exprs: Vec<Expr> = kwargs
        .exprs
        .into_iter()
        .map(|expbin| {
            let res: Result<Expr, serde_json::Error> = serde_json::from_str(expbin.as_str());
            match res {
                Ok(expr) => expr,
                Err(e) => {
                    eprintln!("Deserialization error: {:?}", e);
                    col("a")
                }
            }
        })
        .collect();

    let mut results: Vec<LazyFrame> = vec![];
    let expr_len = exprs.len() - 1;
    for _ in (0..expr_len).step_by(2) {
        let condition = exprs.remove(0);
        let operation = exprs.remove(0);
        let df_partition_col = df
            .clone()
            .lazy()
            .with_columns(vec![condition.alias("___condition")])
            .collect()?;
        let partitions = df_partition_col.partition_by(vec!["___condition"], true)?;
        let (df_true, df_false) = split_parts(&partitions);
        if let Some(df_true_some) = df_true {
            let new_result=df_true_some
            .clone()
            .lazy()
            .select(vec![operation.alias("operation"), col("___row_index")]);
        results.push(new_result);
        }
        match df_false {
            Some(df_false)=>df=df_false,
            None=>df=DataFrame::empty()
        }
    }
    if df.shape().0>0 {
    let operation = exprs.remove(0);
    let def_res = df
        .clone()
        .lazy()
        .select(vec![operation.alias("operation"), col("___row_index")]);
    results.push(def_res);
    }
    let df = concat(results, UnionArgs::default())?
        .sort(vec!["___row_index"], SortMultipleOptions::default())
        .select(vec![col("operation")])
        .collect()?;
    let out_col = df.column("operation")?.clone();
    Ok(out_col.take_materialized_series())
}

fn split_parts(dfs: &[DataFrame]) -> (Option<DataFrame>, Option<DataFrame>) {
    let df1 = dfs[0].clone();
    let df1_condition = df1.column("___condition").unwrap();
    let df1_bool = df1_condition.bool().unwrap().get(0).unwrap();
    match dfs.len() {
        1=> { match df1_bool {
            true=> (Some(df1), None),
            false=>(None, Some(df1))
        }},
        2=> {
            let df2 = dfs[1].clone();
            match df1_bool {
                true=> (Some(df1), Some(df2)),
                false=> (Some(df2), Some(df1))
            }
        },
        _=>panic!("how are there more than 2 partitions?")
    }

}
