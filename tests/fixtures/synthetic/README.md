Synthetic fixtures are generated at test time by `tests/conftest.py`
(`build_synthetic_session`), not stored as static files here — this keeps
the repo free of binary test fixtures while still giving fully offline,
deterministic test data.
