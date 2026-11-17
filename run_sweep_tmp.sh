#!/bin/bash
BIN=/mnt/c/Users/tobia/Desktop/tobifun/sandbox/forks/llama.cpp/build/bin/llama-perplexity
M=/mnt/c/Users/tobia/Desktop/tobifun/sandbox/gemma-4-E2B-it-UD-IQ3_XXS.gguf
F=/tmp/dna_ppl/wikitext-2-raw/wiki.test.raw
export LLAMA_DNA_MAX_BATCH=512
export LLAMA_GRAPH_REUSE_DISABLE=1
export LLAMA_DNA_VERBOSE=1
# CPU overheats: keep threads at 4 (was 8). Points: dense already have; rerun sweep at T=0.0/0.1/0.2/0.3 + zmode check.
for T in 0.0 0.1 0.2 0.3; do
  LLAMA_DNA_T=$T "$BIN" -m "$M" -f "$F" -c 512 --chunks 4 -t 4 --fit off --dnapred iq4_xs > /tmp/sw_iq_T${T}.log 2>&1
done
LLAMA_DNA_T=0.5 LLAMA_DNA_ZMODE=1 "$BIN" -m "$M" -f "$F" -c 512 --chunks 4 -t 4 --fit off --dnapred iq4_xs > /tmp/sw_iq_T0.5_z1.log 2>&1
echo DONE > /tmp/sw_done.txt
