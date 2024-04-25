#!/usr/bin/env bash

set -e
#set -x

DRIVER="$1"

for f in $(find . -name '*.fpcore.c'); do
    ./run-one.sh "$f" "$DRIVER"
done
