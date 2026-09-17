#!/usr/bin/env zsh
# Parallel-chunk downloader with size verification. The HF python client stalled
# repeatedly on this network, so chunks are fetched directly and reassembled.
set -u
repo="$1"; dest="$2"; shift 2

# GNU coreutils and BSD/macOS stat take different flags for "size in bytes".
if stat -c%s . >/dev/null 2>&1; then
  fsize() { stat -c%s "$1"; }
else
  fsize() { stat -f%z "$1"; }
fi

mkdir -p "$dest"
for f in "$@"; do
  url="https://huggingface.co/$repo/resolve/main/$f"
  target="$dest/$f"
  mkdir -p "$(dirname "$target")"
  total=$(curl -sIL "$url" | awk 'BEGIN{IGNORECASE=1}/^content-length:/{v=$2}END{gsub(/\r/,"",v);print v}')
  if [[ -f "$target" && -n "$total" && "$(fsize "$target")" == "$total" ]]; then
    echo "skip $f (already $total bytes)"; continue
  fi
  echo "== $f  ${total} bytes"
  if [[ -z "$total" || "$total" -lt 20000000 ]]; then
    curl -sL --retry 8 --retry-all-errors -o "$target" "$url"
    echo "   -> $(fsize "$target") bytes"
    continue
  fi
  rm -f -- "$target".part*(N)
  parts=6; chunk=$(( total / parts )); pids=()
  for i in $(seq 0 $((parts-1))); do
    start=$(( i * chunk )); end=$(( start + chunk - 1 ))
    [[ $i -eq $((parts-1)) ]] && end=$(( total - 1 ))
    ( curl -sL --retry 8 --retry-all-errors -r "${start}-${end}" -o "$target.part$i" "$url" ) &
    pids+=($!)
  done
  for p in $pids; do wait $p; done
  cat "$target".part* > "$target"; rm -f -- "$target".part*(N)
  got=$(fsize "$target")
  if [[ "$got" == "$total" ]]; then echo "   -> OK $got bytes"; else echo "   -> SIZE MISMATCH got=$got want=$total"; fi
done
