#!/usr/bin/env bash
# Sync forehand clips into /tmp (ephemeral). Override with FOREHAND_S3_URI / FOREHAND_VIDEO_DIR.
set -euo pipefail

S3_URI="${FOREHAND_S3_URI:-s3://tennis-swing-data/Forehands/}"
DEST="${FOREHAND_VIDEO_DIR:-/tmp/Forehands}"

mkdir -p "$DEST"
echo "Syncing ${S3_URI} -> ${DEST}/"
aws s3 sync "$S3_URI" "${DEST}/"
echo "Rear-view clips: ${DEST}/Rear View/"
