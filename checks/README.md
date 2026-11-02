# Behavior checks

Plain scripts that verify execution behavior. Nothing runs them for you: run one by hand when the
code in the second column changes. Numbers are not checked here; they are in `measure/`.

| Script | Run it when you change | What it verifies |
| --- | --- | --- |
| `python checks/terms.py` | a term, the registry, the compose layer | every registered term builds and gives a finite value and gradient |
| `python measure/s6_compose.py` | the compose layer | ranges, weights, monitors, video frames and seeds follow their rules |
