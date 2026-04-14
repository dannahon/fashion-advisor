#!/bin/bash
# Ingest all blog sources into Pinecone, one at a time.
# Run this and let it go — it'll take a few hours due to rate limits.
#
# Usage: cd ~/Desktop/fashion-advisor && bash ingest_all.sh

cd "$(dirname "$0")"

FILES=(
  ~/Downloads/permanentstyle_posts.json
  ~/Downloads/putthison_posts.json
  ~/Downloads/highsnobiety_posts.json
  ~/Downloads/effortlessgent_posts.json
  ~/Downloads/fashionbeans_posts.json
  ~/Downloads/hespokestyle_posts.json
  ~/Downloads/hypebeast_posts.json
  ~/Downloads/menswearmusings_posts.json
  ~/Downloads/thenobledandy_posts.json
  ~/Downloads/opumo_posts.json
  ~/Downloads/stylegirlfriend_posts.json
  ~/Downloads/thevou_posts.json
  ~/Downloads/mensfashionmag_posts.json
  ~/Downloads/dappered_posts.json
)

echo "=========================================="
echo "  Ingesting all blog sources to Pinecone"
echo "=========================================="
echo ""

for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    echo "▶ Starting: $(basename "$f")"
    python3 -u ingest.py "$f"
    echo ""
    echo "⏳ Waiting 10s before next file..."
    sleep 10
  else
    echo "⚠ Skipping: $f (not found)"
  fi
done

echo ""
echo "🎉 All done!"
