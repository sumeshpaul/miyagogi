#!/bin/bash
WATCH_DIR="/mnt/ssd2tb/projects/miyagogi"

echo "🔍 Watching $WATCH_DIR for changes..."

fswatch -o "$WATCH_DIR" | while read; do
  cd "$WATCH_DIR"
  git add .
  git commit -m "Auto sync: $(date '+%Y-%m-%d %H:%M:%S')" > /dev/null 2>&1 || continue
  git push origin main
  echo "✅ Synced at $(date)"
done
