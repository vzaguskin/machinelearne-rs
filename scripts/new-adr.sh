#!/bin/bash
set -e

if [ $# -lt 1 ]; then
  echo "Usage: $0 'short-title-in-kebab-case'"
  exit 1
fi

TITLE="$1"
ADR_DIR="docs/adr"
LAST_NUM=$(ls "$ADR_DIR"/[0-9]*-*.md 2>/dev/null | sed 's/.*\/\([0-9]\+\)-.*/\1/' | sort -n | tail -1)

if [ -z "$LAST_NUM" ]; then
  NEXT_NUM=1
else
  NEXT_NUM=$((LAST_NUM + 1))
fi

PADDED_NUM=$(printf "%04d" $NEXT_NUM)
FILENAME="${ADR_DIR}/${PADDED_NUM}-${TITLE}.md"

if [ -f "$FILENAME" ]; then
  echo "File $FILENAME already exists!"
  exit 1
fi

sed "s/{number}/${NEXT_NUM}/g; s/{short-title}/${TITLE// /-/}/g" "$ADR_DIR/template.md" > "$FILENAME"

echo "Created: $FILENAME"
echo "Now edit it and change 'Status' to 'Accepted' when ready."