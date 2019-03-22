#!/bin/bash

python create_pretraining_data.py --input_file "/datastore/liu121/sentidata2/data/sentences.txt" \
                                  --output_file "/datastore/liu121/sentidata2/data/sent_output.txt" \
                                  --vocab_file "/datastore/liu121/sentidata2/data/sent_vocab.txt"