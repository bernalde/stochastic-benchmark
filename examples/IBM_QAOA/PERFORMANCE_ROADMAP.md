# IBM QAOA Processing Performance Roadmap

This document outlines performance optimizations for `ibm_qaoa_processing.py` to efficiently handle ~1,000 JSON files.

## Context

**Current State:**
- Sequential file processing
- Per-file pickle writes (N disk writes for N files)
- Magic number thresholds (`≤5 rows` for interpolation skip)
- Print-based progress reporting
- Single-trial bootstrap fabrication in main loop

**Target Scale:** ~1,000 JSON files (mostly single-trial)

**Goal:** Reduce wall-clock time and memory footprint without overengineering.

---

## Immediate Changes (Implemented)

### 1. In-Memory Aggregation
**Problem:** Writing intermediate pickles for each file adds ~1-2ms overhead per file (1-2s for 1k files).

**Solution:** Accumulate all parsed trials in memory; write pickles only on explicit request.

**Implementation:**
- Introduce `ProcessingConfig(persist_raw=False)` flag
- Skip `prepare_stochastic_benchmark_data` call unless `persist_raw=True`
- Build `agg_df` directly from accumulated rows

**Impact:** Eliminates ~1-2s file I/O overhead; reduces disk clutter.

**Testing:** Modify `test_process_creates_pickles` to use `ProcessingConfig(persist_raw=True)`.

---

### 2. ProcessingConfig Dataclass
**Problem:** Hardcoded magic numbers scattered throughout code (interpolation threshold, seed, flags).

**Solution:** Centralize configuration in a typed dataclass.

**Schema:**
```python
@dataclass
class ProcessingConfig:
    persist_raw: bool = False                    # Write per-file pickles
    interpolate_diversity_threshold: int = 3     # Skip if instances*depths < threshold
    fabricate_single_trial: bool = True          # Create synthetic bootstrap for single trials
    seed: int = 42                               # Random seed for train/test split
    log_progress_interval: int = 50              # Log every N files
```

**Impact:** Improves maintainability; enables easy tuning without code edits.

**Testing:** Add test verifying config defaults and overrides.

---

### 3. Diversity-Based Interpolation Heuristic
**Problem:** Current heuristic (`n_rows <= 5`) conflates data volume with structural diversity.

**Solution:** Compute `diversity = unique_instances * unique_depths`; skip interpolation if below threshold.

**Logic:**
- 1 instance × 5 depths = 5 → diversity=5 (skip if threshold=6)
- 5 instances × 1 depth = 5 → diversity=5 (skip if threshold=6)
- 2 instances × 3 depths = 6 → diversity=6 (attempt interpolation)

**Implementation:**
```python
unique_instances = combined_bs_results['instance'].nunique()
unique_depths = combined_bs_results['p'].nunique() if 'p' in combined_bs_results else 1
diversity = unique_instances * unique_depths

if diversity < config.interpolate_diversity_threshold:
    logger.info(f"Skipping interpolation: diversity={diversity} < threshold={config.interpolate_diversity_threshold}")
    # fallback to bootstrap results
```

**Impact:** More accurate interpolation decision; prevents spurious skips for sparse multi-instance data.

**Testing:** Add test with synthetic data: 2 instances × 1 depth (skip), 2 × 2 (run).

---

### 4. Structured Logging
**Problem:** `print()` statements lack timestamps, severity, and filtering control.

**Solution:** Replace with `logging` module.

**Levels:**
- **INFO:** Files found, progress every N files, interpolation decision, bootstrap summary
- **WARNING:** Malformed JSON files skipped
- **DEBUG:** Per-file parse details (disabled by default)

**Implementation:**
```python
import logging

logger = logging.getLogger(__name__)

# In process_qaoa_data
logger.info(f"Found {len(json_files)} JSON files.")
for i, json_file in enumerate(json_files):
    if (i + 1) % config.log_progress_interval == 0:
        logger.info(f"Processed {i+1}/{len(json_files)} files...")
```

**Backward Compatibility:** Check if root logger configured; if not, add basic handler to preserve print-like behavior.

**Impact:** Production-ready traceability; easy integration with external logging infrastructure.

**Testing:** Capture log output and assert presence of expected messages.

---

## Deferred Enhancements (Future Implementation)

### 5. Parallel I/O (Medium Impact, Medium Complexity)
**Opportunity:** JSON loading is I/O-bound; threading can parallelize disk reads.

**Approach:**
```python
from concurrent.futures import ThreadPoolExecutor

def parse_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    # ... filename parsing, conversion
    return rows  # list of dicts

if len(json_files) > config.parallel_threshold:
    with ThreadPoolExecutor(max_workers=8) as executor:
        all_rows = list(executor.map(parse_file, json_files))
    # flatten and build DataFrame
```

**Risks:**
- GIL contention if JSON parsing is CPU-bound (unlikely for small files)
- Complexity in error handling (need to aggregate failures)

**Estimated Impact:** 2-3× speedup for I/O-bound phase on multi-core systems.

**Threshold:** Enable only if `len(json_files) > 200` to avoid thread overhead for small runs.

---

### 6. Columnar Output Format (Low Impact, Medium Complexity)
**Opportunity:** Parquet provides better compression and faster read for iterative analysis.

**Approach:**
```python
if config.output_format == 'parquet':
    agg_df.to_parquet(
        os.path.join(output_dir, 'qaoa_results.parquet'),
        compression='snappy',
        index=False
    )
```

**Transition Strategy:**
- Dual-write mode: generate both pickle and parquet initially
- Switch notebooks to parquet after validation period
- Deprecate pickles once stable

**Dependencies:** `pyarrow` or `fastparquet`

**Estimated Impact:** ~50% smaller disk footprint; 2-5× faster load for repeated analysis.

---

### 7. Fast JSON Parser (Low Impact, Low Complexity)
**Opportunity:** `orjson` is 2-3× faster than stdlib `json` for parsing.

**Approach:**
```python
try:
    import orjson
    def load_json(path):
        with open(path, 'rb') as f:
            return orjson.loads(f.read())
except ImportError:
    import json
    def load_json(path):
        with open(path, 'r') as f:
            return json.load(f)
```

**Estimated Impact:** 5-10% overall speedup (JSON parse is small fraction of total time).

**Testing:** Mock `orjson` availability; verify both code paths work.

---

### 8. Lazy Single-Trial Bootstrap Fabrication (Low Impact, Medium Complexity)
**Opportunity:** Defer fabrication of 5 synthetic bootstrap rows until after main bootstrap attempt.

**Current Behavior:** Loop over data_files, fabricate rows for single-trial instances immediately.

**Proposed Behavior:**
1. During ingestion, tag `(instance_id, p)` pairs with `trial_count=1`
2. After main bootstrap, check which instances lack bootstrap results
3. Fabricate only for those instances

**Estimated Impact:** Minor (already in-memory); primarily reduces code duplication.

---

### 9. Memory Optimization: Categorical Dtypes (Low Impact, Low Complexity)
**Opportunity:** Low-cardinality string columns (`trainer`, `evaluator`) consume 8-16× more memory than categorical encoding.

**Approach:**
```python
for col in ['trainer', 'evaluator']:
    if col in agg_df.columns:
        agg_df[col] = agg_df[col].astype('category')
```

**Estimated Impact:** 10-20% memory reduction for large DataFrames (diminishing for 1k files).

---

### 10. Rich Diversity Metric (Low Impact, High Complexity)
**Opportunity:** Current diversity = `instances * depths` ignores trial distribution.

**Advanced Metric:**
```python
diversity = (unique_instances * unique_depths) * log(median_trials_per_group + 1)
```

**Use Case:** Distinguish between:
- 10 instances × 1 depth × 1 trial each (low diversity, skip interpolation)
- 10 instances × 1 depth × 100 trials each (high diversity, attempt interpolation)

**Recommendation:** Defer until current heuristic proves insufficient in practice.

---

## Implementation Order

**Phase 1 (Now):**
1. ✅ ProcessingConfig dataclass
2. ✅ In-memory aggregation
3. ✅ Diversity-based interpolation heuristic
4. ✅ Structured logging

**Phase 2 (If Profiling Shows Need):**
5. Parallel I/O (if per-file time > 10ms)
6. Fast JSON parser (if JSON parse > 20% of runtime)
7. Categorical dtypes (if memory > 500MB for 1k files)

**Phase 3 (If Scale Increases):**
8. Parquet output (if repeated analysis becomes bottleneck)
9. Lazy bootstrap fabrication (if fabrication time > 5% of total)
10. Rich diversity metric (if current heuristic causes issues)

---

## Performance Targets

**Baseline (Current):** ~20-30s for 1,000 files (estimated)

**After Phase 1:** <15s (50% improvement from aggregation + logging overhead reduction)

**After Phase 2:** <10s (additional 30% from parallelization, if enabled)

---

## Validation & Metrics

**Key Metrics:**
- **Wall-clock time:** Total runtime from file discovery to final `sb, agg_df` return
- **Files/sec:** Throughput (target: >50 files/sec for Phase 1)
- **Peak memory:** RSS during aggregation (target: <500MB for 1k files)
- **Interpolation accuracy:** Manual spot-check of diversity decision correctness

**Profiling:**
```bash
python -m cProfile -o profile.stats ibm_qaoa_processing.py
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative').print_stats(20)"
```

**Memory:**
```python
import tracemalloc
tracemalloc.start()
# ... run processing
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory: {peak / 1024 / 1024:.1f} MB")
```

---

## Testing Strategy

**New Tests:**
1. `test_processing_config_defaults` - Verify default values
2. `test_processing_config_overrides` - Pass custom config
3. `test_persist_raw_flag` - Pickles created only when `persist_raw=True`
4. `test_diversity_heuristic` - 2×1, 2×2, 1×5 cases
5. `test_logging_output` - Capture logs, assert messages present
6. `test_large_batch` - Synthetic 200-file run, assert runtime < threshold

**Updated Tests:**
- `test_process_creates_pickles` → Add `ProcessingConfig(persist_raw=True)`

---

## Backward Compatibility

**Flags for Gradual Rollout:**
- `persist_raw=False` default (opt-in for legacy pickle behavior)
- Logger checks for existing configuration before adding handlers
- Config parameter optional (defaults instantiated if not provided)

**Migration Path:**
1. Deploy with default config (minimal behavioral change)
2. Monitor logs for interpolation decisions
3. Tune `interpolate_diversity_threshold` based on actual data patterns
4. Enable parallelization via config flag once validated

---

## Summary

**Immediate wins (Phase 1):**
- Remove 1-2s of unnecessary disk I/O
- Clearer configuration management
- More accurate interpolation logic
- Production-ready logging

**Future optimizations (Phase 2+):**
- Parallelization for 2-3× speedup if needed
- Parquet for storage efficiency
- Memory tuning for larger datasets

**Trade-offs:**
- Prioritize simplicity over premature optimization
- Profile before implementing Phase 2
- Keep all enhancements behind config flags for safe rollout
