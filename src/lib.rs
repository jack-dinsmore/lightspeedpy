use pyo3::prelude::*;

/// Doubles a number, from Rust
#[pyfunction]
fn double(x: i32) -> PyResult<i32> {
    Ok(x * 2)
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(double, m)?)?;
    Ok(())
}
