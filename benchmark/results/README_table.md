| model | condition | accuracy | target first | confusable acc. | no tool | cut off | tools/req | input tok/req | list $/1k | billed $/1k | p50 latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DeepSeek V4 Flash | all tools (10) | 99.0% | 97.9% | 100.0% | 0.0% | 0.0% | 10 | 3,461 | 0.30 | 0.30 | 3.9s |
| DeepSeek V4 Flash | all tools (25) | 95.9% | 94.8% | 96.5% | 2.1% | 2.1% | 25 | 8,307 | 0.68 | 0.67 | 3.7s |
| DeepSeek V4 Flash | all tools (50) | 97.9% | 96.9% | 100.0% | 1.0% | 1.0% | 50 | 16,219 | 1.30 | 1.28 | 3.9s |
| DeepSeek V4 Flash | all tools (100) | 93.8% | 92.8% | 100.0% | 1.0% | 0.0% | 100 | 31,436 | 2.50 | 2.45 | 5.3s |
| DeepSeek V4 Flash | all tools (200) | 91.8% | 90.7% | 89.7% | 0.0% | 0.0% | 200 | 62,793 | 4.96 | 4.87 | 6.3s |
| DeepSeek V4 Flash | all tools (454) | 87.6% | 86.6% | 89.7% | 0.0% | 0.0% | 454 | 143,685 | 11.31 | 2.46 | 8.6s |
| DeepSeek V4 Flash | toolmem top-5 | 66.0% | 62.9% | 58.6% | 16.5% | 2.1% | 5 | 2,279 | 0.21 | 0.21 | 4.3s |
| DeepSeek V4 Flash | toolmem top-5 + augmented | 81.4% | 78.3% | 75.9% | 11.3% | 1.0% | 5 | 2,049 | 0.19 | 0.19 | 4.1s |
| DeepSeek V4 Flash | toolmem top-10 + augmented | 86.6% | 84.5% | 82.8% | 4.1% | 2.1% | 10 | 3,989 | 0.35 | 0.34 | 2.4s |
| DeepSeek V4 Flash | toolmem top-20 + augmented | 92.8% | 90.7% | 93.1% | 1.0% | 1.0% | 20 | 7,955 | 0.66 | 0.42 | 2.5s |
| DeepSeek V4 Flash | toolmem top-50 + augmented | 86.6% | 85.6% | 89.7% | 1.0% | 1.0% | 50 | 19,139 | 1.53 | 0.58 | 4.5s |
| DeepSeek V4 Flash | toolmem top-20 + augmented, shuffled | 84.5% | 83.5% | 86.2% | 5.1% | 5.1% | 20 | 7,843 | 0.66 | 0.10 | 10.3s |
| Qwen3.7 Flash | all tools (10) | 99.0% | 97.9% | 96.5% | 1.0% | 0.0% | 10 | 3,465 | 0.13 | 0.13 | 2.8s |
| Qwen3.7 Flash | all tools (25) | 99.0% | 97.9% | 100.0% | 0.0% | 0.0% | 25 | 8,318 | 0.28 | 0.28 | 2.8s |
| Qwen3.7 Flash | all tools (50) | 97.9% | 96.9% | 96.5% | 0.0% | 0.0% | 50 | 16,269 | 0.52 | 0.52 | 3.0s |
| Qwen3.7 Flash | all tools (100) | 94.8% | 93.8% | 93.1% | 2.1% | 1.0% | 100 | 31,552 | 0.98 | 1.88 | 3.9s |
| Qwen3.7 Flash | all tools (200) | 86.6% | 85.6% | 82.8% | 0.0% | 0.0% | 200 | 63,028 | 1.92 | 6.39 | 6.5s |
| Qwen3.7 Flash | all tools (454) | 86.6% | 85.6% | 89.7% | 2.1% | 1.0% | 454 | 144,231 | 4.35 | 4.65 | 2.5s |
| Qwen3.7 Flash | toolmem top-5 | 67.0% | 63.9% | 58.6% | 17.5% | 2.1% | 5 | 2,265 | 0.11 | 0.11 | 2.8s |
| Qwen3.7 Flash | toolmem top-5 + augmented | 82.5% | 79.4% | 75.9% | 10.3% | 0.0% | 5 | 2,038 | 0.10 | 0.10 | 2.8s |
| Qwen3.7 Flash | toolmem top-10 + augmented | 86.6% | 84.5% | 82.8% | 4.1% | 0.0% | 10 | 3,918 | 0.15 | 0.15 | 2.4s |
| Qwen3.7 Flash | toolmem top-20 + augmented | 89.7% | 88.7% | 86.2% | 0.0% | 0.0% | 20 | 7,878 | 0.27 | 0.26 | 2.6s |
| Qwen3.7 Flash | toolmem top-50 + augmented | 89.7% | 88.7% | 89.7% | 0.0% | 0.0% | 50 | 19,045 | 0.60 | 0.65 | 3.1s |
| Qwen3.7 Flash | toolmem top-20 + augmented, shuffled | 90.7% | 89.7% | 93.1% | 1.0% | 0.0% | 20 | 7,878 | 0.27 | 0.27 | 2.8s |
| Jev | all tools (10) | 100.0% | 97.9% | 100.0% | 0.0% | 0.0% | 10 | 987 | 0.04 | 0.04 | 0.5s |
| Jev | all tools (25) | 99.0% | 96.9% | 100.0% | 0.0% | 0.0% | 25 | 1,924 | 0.08 | 0.08 | 0.5s |
| Jev | all tools (50) | 99.0% | 94.8% | 96.5% | 0.0% | 0.0% | 50 | 3,496 | 0.15 | 0.15 | 0.5s |
| Jev | all tools (100) | 99.0% | 94.8% | 96.5% | 0.0% | 0.0% | 100 | 6,631 | 0.28 | 0.28 | 0.5s |
| Jev | all tools (200) | 97.9% | 92.8% | 96.5% | 0.0% | 0.0% | 200 | 12,875 | 0.54 | 0.54 | 0.6s |
| Jev | toolmem top-10 + augmented | 88.7% | 86.6% | 86.2% | 5.1% | 0.0% | 10 | 1,027 | 0.04 | 0.04 | 0.5s |
| Jev | toolmem top-20 + augmented | 93.8% | 92.8% | 93.1% | 2.1% | 0.0% | 20 | 1,713 | 0.07 | 0.07 | 0.5s |
| Jev | toolmem top-50 + augmented | 95.9% | 93.8% | 93.1% | 0.0% | 0.0% | 50 | 3,749 | 0.16 | 0.16 | 0.5s |
| Jev | toolmem top-20 + augmented, shuffled | 95.9% | 92.8% | 96.5% | 1.0% | 0.0% | 20 | 1,713 | 0.07 | 0.07 | 0.5s |
| Jev, server then tool | all tools (454) | 92.8% | 89.7% | 86.2% | 1.0% | 0.0% | 454 | 4,667 | 0.20 | 0.20 | 0.8s |
| Jev, then DeepSeek if unsure | toolmem top-20 + augmented | 93.8% | 92.8% | 93.1% | 0.0% | 0.0% | 20 | 2,514 | 0.16 | 0.16 | 0.5s |
| Jev, then Qwen if unsure | toolmem top-20 + augmented | 93.8% | 92.8% | 89.7% | 1.0% | 0.0% | 20 | 2,454 | 0.11 | 0.11 | 0.7s |

Retrieval only (does the expected tool appear in the top-k?):

| embedder | mode | hit@1 | hit@3 | hit@5 | hit@10 | MRR |
|---|---|---:|---:|---:|---:|---:|
| openai:text-embedding-3-small | keyword | 29.9% | 51.5% | 59.8% | 68.0% | 0.426 |
| openai:text-embedding-3-small | semantic | 36.1% | 57.7% | 65.0% | 82.5% | 0.498 |
| openai:text-embedding-3-small | hybrid | 33.0% | 63.9% | 70.1% | 76.3% | 0.493 |
| openai:text-embedding-3-small | hybrid+aug | 57.7% | 81.4% | 86.6% | 91.8% | 0.706 |

454 tools, 97 queries, embedder `openai:text-embedding-3-small`.
