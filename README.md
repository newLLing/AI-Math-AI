# AI-Math-AI

Algorithm implementations, search code, and archived experiment records.

## Contents

- `algorithm/`: RARS, Leverage, and variance-scaled FPS implementations.
- `lmms-eval/`: AutoResearch orchestration, exploration, coding, and task code.
- `records/run_records.zip`: experiment iterations, generated candidates, logs, and evaluation records.
- `records/historical_snapshot.zip`: historical files retained separately from the main experiment records.
- `MANIFEST.json` and `SANITIZATION_REPORT.json`: provenance metadata supplied with the source export.
- `PACKAGING.json`: archive sizes, SHA-256 checksums, and packaging verification results.

## Restore experiment records

Run from the repository root:

```bash
unzip records/run_records.zip -d .
unzip records/historical_snapshot.zip -d .
```

The archives retain the original relative paths. Extracted records are ignored by Git to avoid committing duplicate copies. Every source file is included either directly or in one archive, and archive contents were checked against the source file hashes before upload.

This is a source and experiment-record export. The supplied dependency and packaging files are retained as provided; a clean installation and full model evaluation have not been validated for this export. Model weights and datasets are not bundled.

## License

See `LICENSE`. Existing source notices are retained.
