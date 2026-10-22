# Tests

Run commands from the repository root with the `d4-gen` environment active.

Run all automated checks:

```bash
python -m unittest discover -s tests -t . -v
```

Run one suite:

```bash
python -m unittest tests.test_dataset_integrity -v
python -m unittest tests.test_generated_distributions -v
python -m unittest tests.test_three_representations -v
python -m unittest tests.test_distribution_scores -v
python -m unittest tests.test_distribution_statistics -v
```

`test_distributed_training.py` is a manual two-node GPU training demo. It is
imported without launching training during discovery. Run it explicitly only
in the intended distributed environment:

```bash
python -m tests.test_distributed_training
```

The four `test_*.ipynb` notebooks contain exploratory checks. Their first cell
sets the working directory and Python import path to the repository root so
existing `datasets/` and `checkpoints/` paths retain their meaning. Run that
cell before the existing notebook cells. The notebooks are not executed by
the automated test suite.
