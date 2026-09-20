# Serialization Format Comparison — Module 1, Step 4

## Format Comparison

| Format   | Human-Readable | Cross-Language | Schema-Enforced | Safe (Untrusted Source) |
|----------|:---:|:---:|:---:|:---:|
| JSON     | Yes | Yes | Yes | Yes |
| Protobuf | No  | Yes | Yes | Yes |
| Pickle   | No  | No  | No  | No  |
| ONNX     | No  | Yes | Yes | Yes |

## Decision

This service serves predictions via ONNX in production. Pickle executes
arbitrary code on load and must never be loaded from an untrusted source;
it is used here only for local training/development, never for serving.

## Parity Test (500 validation rows)

- Max absolute difference (pickle vs ONNX): 0.00000417
- Threshold: atol=1e-4
- Result: PASSED

## Latency Benchmark (500 rows, 10 runs each)

| Format | Mean | p95 |
|--------|------|-----|
| Pickle | 5.633ms | 7.529ms |
| ONNX   | 11.439ms | 28.991ms |

Note: ONNX appears slower here because predict_onnx() creates a new
InferenceSession (loading the model from disk) on every call. In real
serving (Step 5's API), the ONNX session is loaded once at startup and
reused for every request - the same "load once, not per-request" principle
the handbook requires for the pickle model.
