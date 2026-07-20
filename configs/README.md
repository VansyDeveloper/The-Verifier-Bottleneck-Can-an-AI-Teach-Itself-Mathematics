# Experiment configurations

Each `cells_*.txt` file describes a batch of training cells consumed by the
phase/chain shell runners. Columns are:

```text
tag temperature generations alpha beta [extra training arguments]
```

These files are intentionally versioned: they are small and form part of the
experiment specification.
