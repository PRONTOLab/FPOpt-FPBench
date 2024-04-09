#!/usr/bin/env sh


SRC="$1"

echo Handling "$SRC"

NEWDRIVER="$SRC.driver.cpp"
cp driver.cpp "$NEWDRIVER"

EXE="$SRC.exe"

NUM="$(grep -e ' ex[0-9]*(' -- $SRC | wc -l)"

echo "int main() {" >> "$NEWDRIVER"
echo "    init();" >> "$NEWDRIVER"
for i in $(seq 0 $(( $NUM - 1 ))); do
    echo "    RUN(ex$i)" >> "$NEWDRIVER"
done
echo "    return 0;" >> "$NEWDRIVER"
echo "}" >> "$NEWDRIVER"

clang++ -Wall "$NEWDRIVER" -g -include "$SRC" -o "$EXE" $CXXFLAGS 1>2

"$EXE"
