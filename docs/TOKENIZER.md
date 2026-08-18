# MiniFrontier V1 tokenizer contract

The V1 tokenizer is byte-level BPE with a requested vocabulary size of 16,384. It uses no Unicode normalization, no unknown token, and no implicit prefix space. The full byte alphabet is present at training time, so arbitrary UTF-8 input is representable.

Special tokens are reserved before BPE training and their IDs are immutable:

| ID | Token |
| ---: | --- |
| 0 | `<|pad|>` |
| 1 | `<|bos|>` |
| 2 | `<|eos|>` |
| 3 | `<|system|>` |
| 4 | `<|user|>` |
| 5 | `<|assistant|>` |
| 6 | `<|fim_prefix|>` |
| 7 | `<|fim_suffix|>` |
| 8 | `<|fim_middle|>` |
| 9 | `<|tool_call|>` |
| 10 | `<|tool_result|>` |

Documents are encoded without an automatic BOS and with one EOS appended by the packing pipeline. BOS is added explicitly for generation/evaluation contexts that require it. PAD is used only for bounded partial batches and never treated as document content. Chat, FIM, and tool placeholder tokens are reserved now even where their V1 behavior is implemented later.

`tokenizer.json` is accompanied by `tokenizer_config.json`, which records the contract version, requested and actual vocabulary size, fixed IDs, and SHA-256 of the tokenizer artifact. Changing token strings, IDs, byte-level settings, or normalization requires a new tokenizer/model lineage rather than an in-place V1 edit.

A standalone tokenizer artifact defaults its advertised model length to 2,048, while a model release
writes the actual preset's `max_seq_len` into `tokenizer_config.json` (for example, 1,024 for the
50M preset). This metadata does not change token IDs or the tokenizer lineage.
