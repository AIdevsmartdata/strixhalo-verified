# Third-party notices

This repository is MIT-licensed (see `LICENSE`). The runtime it builds is a fork of llama.cpp/ggml (MIT, "Copyright (c)
2023-2026 The ggml authors"), through [Aristo94/GSQHalo.cpp](https://github.com/Aristo94/GSQHalo.cpp) (MIT) and with pull
requests from [halo-box/strix-llama.cpp](https://github.com/halo-box/strix-llama.cpp) (MIT). Their licence texts ship with
the source tree that `runtime/build.sh` checks out.

## Gufo (MIT)

Parts of the patch in `runtime/patches/` adapt code or ideas from [gufo-org/gufo](https://github.com/gufo-org/gufo):
exact speculative sampling of MTP drafts (`common/leviathan.h`), copy-lookup drafting rules (`common/copie24.h`), the
draft-length policy (`common/draft-policy*.h`), and several HIP kernel techniques (cross-lane reductions, vector loads,
early loads, Q8 batch launch, the QSA decode schedule). Each of those files carries Gufo's copyright line. Gufo's licence:

```
MIT License

Copyright (c) 2026 gufo contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Data files

- `launch/mtp_vocab_65k_fr4096.bin`: 69,632 token ids (uint32) of the Qwen3.8-Flash-Next tokenizer. It holds a
  frequency-ranked 65k set (the "frspec" idea of drluoto/Qwen3.8-Flash-Next-MTP-GGUF) plus the 4,096 most frequent
  missing French tokens, counted on Project Gutenberg public-domain books (Hugo, Zola) and French Wikipedia extracts.
  No text is included, only token ids.
- Model weights are not distributed here. Qwen3.8-Flash-Next is under the Qwen community licence; see the model card.
