#!/usr/bin/env bash

set -e
#set -x


for f in $(find . -name '*.fpcore.c'); do
    ./run-one.sh "$f"
done
