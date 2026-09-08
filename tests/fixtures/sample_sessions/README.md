# Real sample sessions

This directory is for real, downloaded sample scan data used for manual/
`realdata`-marked validation. It is gitignored (except this file) — nothing
here is committed.

To fetch a real sample:

```bash
python scripts/download_sample_data.py --dataset 3dscannerapp_samples --output data/samples
```

Then ingest a scan from the downloaded repo into this directory, e.g.:

```bash
spatial-os ingest data/samples/3dscannerapp_samples/<example_dir> \
  --format 3dscannerapp \
  --output tests/fixtures/sample_sessions/<example_dir>
```

Unit and e2e tests never require this directory to be populated — they run
against the procedurally generated fixture in `tests/fixtures/synthetic/`
(see `tests/conftest.py`). Tests marked `@pytest.mark.realdata` are skipped
unless data is present here.
