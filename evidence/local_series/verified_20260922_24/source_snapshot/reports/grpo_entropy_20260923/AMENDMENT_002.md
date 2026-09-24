# Corrected base-model hash before any entropy-ablation training

The second technical benchmark stopped at frozen-input validation, before
loading tasks or executing an optimizer step. Its command and protocol text
had a 63-character transcription of the base-model SHA-256. An independent
read inside the Docker environment confirmed the actual 64-character digest
`CD2A512003E2F9F3CD3C32A9C3573F820BB28C940F73C57B1DDAA983D9223EBA`.
The training-task and atomic-adapter hashes matched. The protocol text is
corrected. The failed benchmark container and `FAILED.json` remain intact.
No reward, held-out score, or pair outcome was inspected before this correction.
