# Run unit tests

for the developers, execute the following command under this directory:

```bash
pytest -v
```

The results should be something like:

```bash


recommenders/test_mmr.py::TestMaximalMarginalRelevance::test_rerank_scale PASSED    [ 33%]

recommenders/test_mmr.py::TestMaximalMarginalRelevance::test_rerank_lambda PASSED    [ 66%]

recommenders/test_mmr.py::TestMaximalMarginalRelevance::test_domain FAILED    [100%]

===================== FAILURES ==========================
________ TestMaximalMarginalRelevance.test_domain ______
```
