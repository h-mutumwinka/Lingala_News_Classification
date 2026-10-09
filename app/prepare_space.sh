#!/usr/bin/env bash
# Copies everything the Hugging Face Space needs into space_upload/.
# Usage (from the project root):  bash app/prepare_space.sh
set -euo pipefail
rm -rf space_upload && mkdir space_upload
cp app/app.py app/examples.json app/requirements.txt app/README.md space_upload/
cp models/tfidf_classifier.joblib space_upload/
echo "Upload the contents of space_upload/ to your Space (web UI: Files > Add file > Upload files)."
