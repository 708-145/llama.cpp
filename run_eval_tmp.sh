#!/bin/bash
# Perplexity sweep driver. Kill procedures and cooling rules: see DNA.md
# "Running perplexity sweeps (WSL / setsid shell)". Use -t 4 and sleep 120
# between CPU-intensive runs to avoid overheating (documented 2026-11-16).
BIN=/mnt/c/Users/tobia/Desktop/tobifun/sandbox/forks/llama.cpp/build/bin/llama-perplexity
M=/mnt/c/Users/tobia/Desktop/tobifun/sandbox/gemma-4-E2B-it-UD-IQ3_XXS.gguf
F=/tmp/dna_ppl/wikitext-2-raw/wiki.test.raw
export LLAMA_DNA_MAX_BATCH=512
export LLAMA_GRAPH_REUSE_DISABLE=1
export LLAMA_DNA_VERBOSE=1
"$BIN" -m "$M" -f "$F" -c 512 --chunks 4 -t 4 --fit off > /tmp/gem_dense4.log 2>&1
sleep 120
LLAMA_DNA_CALIB=4096 LLAMA_DNA_T=0.1 "$BIN" -m "$M" -f "$F" -c 512 --chunks 4 -t 4 --fit off --dnapred iq4_xs > /tmp/gem_calib_T0.1.log 2>&1
sleep 120
LLAMA_DNA_CALIB=4096 LLAMA_DNA_T=0.5 "$BIN" -m "$M" -f "$F" -c 512 --chunks 4 -t 4 --fit off --dnapred iq4_xs > /tmp/gem_calib_T0.5.log 2>&1
echo DONE > /tmp/gem_calib_done.txt
