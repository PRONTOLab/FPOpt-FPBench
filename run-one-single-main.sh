#!/usr/bin/env sh

set -e

SRC="$1"
DRIVER="$2"

echo Handling "$SRC"

NEWDRIVER="$SRC.driver.cpp"
AOUT="$SRC.a.out"

NUM="$(grep -e ' ex[0-9]*(' -- $SRC | wc -l)"

cp "$DRIVER" "$NEWDRIVER"
echo "int main() {" >> "$NEWDRIVER"
echo "    init();" >> "$NEWDRIVER"

for i in $(seq 0 $(( $NUM - 1 ))); do
    echo "    RUN(ex$i);" >> "$NEWDRIVER"
done

echo "    cleanup();" >> "$NEWDRIVER"
echo "    return 0;" >> "$NEWDRIVER"
echo "}" >> "$NEWDRIVER"

clang++ -Wall "$NEWDRIVER" -O3 -include "$SRC" -o "$AOUT" $CXXFLAGS 1>2
set +e
timeout -k 1s 5s "$AOUT"
TIMEOUT_RESULT=$?
set -e

if [ $TIMEOUT_RESULT -eq 124 ]; then
    echo Timed out.
elif [ $TIMEOUT_RESULT -ne 0 ]; then
    echo Failed.
    false
fi
