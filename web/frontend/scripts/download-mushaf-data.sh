#!/bin/bash
# Download mushaf layout data (604 JSON files) and QPC V1 WOFF2 fonts
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRONTEND_DIR="$(dirname "$SCRIPT_DIR")"
PUBLIC_DIR="$FRONTEND_DIR/public"

LAYOUT_DIR="$PUBLIC_DIR/mushaf-layout"
FONT_DIR="$PUBLIC_DIR/fonts/qpc-v1"

mkdir -p "$LAYOUT_DIR" "$FONT_DIR"

LAYOUT_BASE="https://raw.githubusercontent.com/zonetecde/mushaf-layout/main/mushaf"
FONT_BASE="https://raw.githubusercontent.com/nuqayah/qpc-fonts/master/mushaf-woff2"

echo "=== Downloading 604 mushaf layout JSONs ==="
for i in $(seq 1 604); do
  padded=$(printf "%03d" $i)
  outfile="$LAYOUT_DIR/page-${padded}.json"
  if [ ! -f "$outfile" ]; then
    curl -sL "${LAYOUT_BASE}/page-${padded}.json" -o "$outfile" &
  fi
  # Limit concurrent downloads
  if (( i % 20 == 0 )); then
    wait
    echo "  Downloaded $i / 604 layout files..."
  fi
done
wait
echo "  Done: 604 layout files"

echo "=== Downloading QPC V1 WOFF2 fonts ==="
# Download bismillah font
if [ ! -f "$FONT_DIR/QCF_BSML.woff2" ]; then
  curl -sL "${FONT_BASE}/QCF_BSML.woff2" -o "$FONT_DIR/QCF_BSML.woff2" &
fi

for i in $(seq 1 604); do
  padded=$(printf "%03d" $i)
  outfile="$FONT_DIR/QCF_P${padded}.woff2"
  if [ ! -f "$outfile" ]; then
    curl -sL "${FONT_BASE}/QCF_P${padded}.woff2" -o "$outfile" &
  fi
  if (( i % 20 == 0 )); then
    wait
    echo "  Downloaded $i / 604 font files..."
  fi
done
wait
echo "  Done: 604 font files + bismillah"

echo ""
echo "=== Combining layout JSONs into single mushaf-pages.json ==="
node -e "
const fs = require('fs');
const path = require('path');
const layoutDir = '$LAYOUT_DIR';
const pages = [];
for (let i = 1; i <= 604; i++) {
  const padded = String(i).padStart(3, '0');
  const filePath = path.join(layoutDir, 'page-' + padded + '.json');
  const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  pages.push(data);
}
fs.writeFileSync(
  path.join('$PUBLIC_DIR', 'mushaf-pages.json'),
  JSON.stringify(pages)
);
console.log('  Combined ' + pages.length + ' pages into mushaf-pages.json');
console.log('  File size: ' + (fs.statSync(path.join('$PUBLIC_DIR', 'mushaf-pages.json')).size / 1024 / 1024).toFixed(1) + ' MB');
"

echo ""
echo "=== All done! ==="
