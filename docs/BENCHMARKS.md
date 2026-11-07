
# Leakguard Runtime and Memory Benchmarks

## Purpose

Leakguard includes a reproducible benchmark for two parts of its
research pipeline:

1. Synthetic order-book data generation.
2. Causal microstructure feature construction.

The benchmark measures runtime and memory indicators at multiple
input sizes. Its purpose is to make engineering costs visible and
provide a repeatable baseline for future optimization.

## Run the benchmark

From the repository root:

```bash
python -m leakguard.benchmark
```

The default row counts are:

- 10,000
- 30,000
- 60,000

To select different sizes:

```bash
python -m leakguard.benchmark --rows 5000 20000 100000
```

To select a random seed:

```bash
python -m leakguard.benchmark --rows 10000 30000 --seed 11
```

The benchmark writes:

```text
results/benchmarks/
    latest.json
    latest.md
```

## Measurements

For each input size, Leakguard records:

- Synthetic data generation wall time.
- Feature construction wall time.
- Python-tracked peak memory during generation.
- Python-tracked peak memory during feature construction.
- Input DataFrame memory.
- Feature DataFrame memory.

Wall time is measured with `time.perf_counter`.

Peak traced memory is measured with Python's `tracemalloc`.
DataFrame payload memory is measured with pandas
`memory_usage(deep=True)`.

## Important interpretation limits

These measurements are not total process resident memory.
They do not include a complete accounting of every native allocation
or temporary operating-system allocation.

The benchmark does not measure:

- Model training or inference.
- Historical data loading throughput.
- Exchange WebSocket throughput.
- Network or exchange latency.
- Order submission or execution.
- Queue position or fill probability.
- Production trading capacity.

Results depend on hardware, software versions, background system
load, and the selected row counts.

The synthetic generator is a controlled software test fixture.
Its performance is not evidence of market-data processing capacity.

## Reporting standard

Benchmark results should be reported with:

- The tested row counts.
- The random seed.
- Python and dependency versions.
- The execution environment.
- The measurement method.
- The limitations above.

Do not compare measurements from different machines as if they
were controlled experiments.

Do not describe these results as production HFT benchmarks.
