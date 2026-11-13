#!/usr/bin/env bash
# Download the emg2pose metadata CSV and the 618 MB mini dataset.
# The full dataset is 431 GB; see the upstream README if you need it.
set -euo pipefail

DEST="${1:-$HOME}"
BASE="https://fb-ctrl-oss.s3.amazonaws.com/emg2pose"

echo "Downloading metadata CSV (5.5 MB) to $DEST ..."
curl -fSL "$BASE/emg2pose_metadata.csv" -o "$DEST/emg2pose_metadata.csv"

if [ ! -d "$DEST/emg2pose_dataset_mini" ]; then
  echo "Downloading mini dataset (618 MB) ..."
  curl -fSL "$BASE/emg2pose_dataset_mini.tar" -o "$DEST/emg2pose_dataset_mini.tar"
  tar -xf "$DEST/emg2pose_dataset_mini.tar" -C "$DEST"
  rm "$DEST/emg2pose_dataset_mini.tar"
else
  echo "Mini dataset already present, skipping."
fi

echo "Done. Data in $DEST/emg2pose_dataset_mini"
