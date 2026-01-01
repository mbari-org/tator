mkdir -p legacy
URL="https://tator-ci.s3.us-east-1.amazonaws.com/bundles_1-3-16.zip"
OUTPUT="/tmp/bundles_1-3-16.zip"

if command -v wget >/dev/null 2>&1; then
    wget -O "$OUTPUT" "$URL"
elif command -v curl >/dev/null 2>&1; then
    curl -o "$OUTPUT" "$URL"
else
    echo "Error: Neither wget nor curl is available" >&2
    exit 1
fi

unzip -o "$OUTPUT" -d "legacy"
