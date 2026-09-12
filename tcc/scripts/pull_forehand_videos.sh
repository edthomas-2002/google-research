#!/usr/bin/env bash
# Sync rear-view forehand clips into /tmp. Override with FOREHAND_S3_URI / FOREHAND_VIDEO_DIR.
set -euo pipefail

S3_URI="${FOREHAND_S3_URI:-s3://tennis-swing-data/Forehands/Rear View/}"
DEST="${FOREHAND_VIDEO_DIR:-/tmp/Forehands/Rear View}"

mkdir -p "$DEST"
echo "Syncing ${S3_URI} -> ${DEST}/"
aws s3 sync "$S3_URI" "${DEST}/"
echo "Rear-view clips: ${DEST}/"
