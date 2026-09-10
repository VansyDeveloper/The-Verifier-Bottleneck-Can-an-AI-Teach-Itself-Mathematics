# Stage 4 compact audit archive v2

This archive contains everything from audit_v1 plus the complete source-code
path used for Stage 4 atomic training, composition distillation, exact ranking,
statistics, release construction, schemas, tests, protocols and preregistration.

Source files retain repository-relative paths below `source/`. See
`audit_v2/SOURCE_CODE_MANIFEST.json` for SHA-256 hashes and the exact inventory.

To keep the archive compact, model weights, adapters/checkpoints, generated
datasets and raw full-ranking shards are intentionally excluded. Those binary
and generated artifacts remain in `stage4_verifier_bottleneck_full_v1.zip`.
The source code is complete; a byte-identical rerun still requires the frozen
model, adapter and data artifacts from the full archive.
