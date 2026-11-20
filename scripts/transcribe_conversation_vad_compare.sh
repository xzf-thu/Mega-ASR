#!/bin/bash

input=$1

python scripts/transcribe_conversation.py --vad simple-vad -i $input -d mps
python scripts/transcribe_conversation.py --vad silero-vad -i $input -d mps
python scripts/transcribe_conversation.py --vad fsmn-vad -i $input -d mps
python scripts/transcribe_conversation.py --vad ten-vad -i $input -d mps