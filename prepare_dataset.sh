python tools/preprocess_data.py \
    --input /mnt/afs/datasets/oscar-1GB.jsonl \
    --output-prefix /mnt/afs/datasets/oscar-1GB \
    --vocab /mnt/afs/tokenizers/gpt2/gpt2-vocab.json \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file /mnt/afs/tokenizers/gpt2/gpt2-merges.txt \
    --append-eod \
    --workers 128