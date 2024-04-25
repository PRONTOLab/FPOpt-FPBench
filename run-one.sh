#!/usr/bin/env sh

set -e

SRC="$1"
DRIVER="$2"

echo Handling "$SRC"

NEWDRIVER="$SRC.driver.cpp"

AOUT="$SRC.a.out"

NUM="$(grep -e ' ex[0-9]*(' -- $SRC | wc -l)"

for i in $(seq 0 $(( $NUM - 1 ))); do
    cp "$DRIVER" "$NEWDRIVER"
    echo "int main() {" >> "$NEWDRIVER"
    echo "    init();" >> "$NEWDRIVER"
    echo "    RUN(ex$i)" >> "$NEWDRIVER"
    echo "    return 0;" >> "$NEWDRIVER"
    echo "}" >> "$NEWDRIVER"
    clang++ -Wall "$NEWDRIVER" -g -include "$SRC" -o "$AOUT" $CXXFLAGS 1>2
    timeout -k 1s 5s "$AOUT" || echo TIMEOUT OR FAIL
done
