//! Data parallelism where there are threads to run it on.
//!
//! The slow parts of growing and baking a tree are made of pieces that do not depend on
//! each other: the stems of a bark sweep, the cells of a leaf atlas, the rows of a mip
//! level. On the desktop they are spread over every core; on the web, which has one
//! thread, the same closures run one after another. Either way the results come back
//! in the order of the input, so what is built does not depend on which ran.

/// `f` over every item, in parallel where possible, with the results in input order.
pub fn map<T, R, F>(items: &[T], f: F) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        items.par_iter().map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        items.iter().map(f).collect()
    }
}

/// `f` over `0..n`, in parallel where possible, with the results in order.
pub fn map_range<R, F>(n: usize, f: F) -> Vec<R>
where
    R: Send,
    F: Fn(usize) -> R + Sync + Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        (0..n).into_par_iter().map(f).collect()
    }
    #[cfg(target_arch = "wasm32")]
    {
        (0..n).map(f).collect()
    }
}

/// `f` over consecutive chunks of `data`, each `chunk` long but the last, with the
/// index of the chunk.
pub fn for_each_chunk_mut<T, F>(data: &mut [T], chunk: usize, f: F)
where
    T: Send,
    F: Fn(usize, &mut [T]) + Sync + Send,
{
    let chunk = chunk.max(1);
    #[cfg(not(target_arch = "wasm32"))]
    {
        use rayon::prelude::*;
        data.par_chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
    }
    #[cfg(target_arch = "wasm32")]
    {
        data.chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
    }
}

/// Two closures, side by side where possible.
pub fn join<A, B, RA, RB>(a: A, b: B) -> (RA, RB)
where
    A: FnOnce() -> RA + Send,
    B: FnOnce() -> RB + Send,
    RA: Send,
    RB: Send,
{
    #[cfg(not(target_arch = "wasm32"))]
    {
        rayon::join(a, b)
    }
    #[cfg(target_arch = "wasm32")]
    {
        (a(), b())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_come_back_in_input_order() {
        let items: Vec<u32> = (0..1000).collect();
        assert_eq!(map(&items, |x| x * 2), items.iter().map(|x| x * 2).collect::<Vec<_>>());
        assert_eq!(map_range(1000, |i| i), (0..1000).collect::<Vec<_>>());
        let mut data = vec![0usize; 1000];
        for_each_chunk_mut(&mut data, 7, |i, c| c.iter_mut().for_each(|v| *v = i));
        assert!(data.iter().enumerate().all(|(k, &v)| v == k / 7));
    }
}
