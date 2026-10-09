#!/usr/bin/env bash
set -euo pipefail

# One-off recovery for Qwen's official AWQ shard when the cloud node cannot
# connect to huggingface.co but can reach its signed CDN URL. Keep the URL in
# MODEL_CDN_URL only; never write it or credentials to the repository.
: "${MODEL_CDN_URL:?Set an expiring official CDN URL in MODEL_CDN_URL}"

model_dir=/root/fsas/vllm-lab/models/Qwen3-8B-AWQ
shard=model-00001-of-00002.safetensors
partial="$model_dir/$shard.incomplete"
final="$model_dir/$shard"
expected_size=4853922024
expected_sha=6e112429856bc65e3837a9f38d6f6b71ffdda832cb46299a12f4fa8f6352516e
parts_dir="$model_dir/.awq_cdn_parts"

[[ -f "$partial" && ! -e "$final" ]] || {
  echo "Expected unfinished shard and no final shard" >&2
  exit 1
}
base_size=$(stat -c %s "$partial")
(( base_size > 0 && base_size < expected_size )) || {
  echo "Unexpected partial size: $base_size" >&2
  exit 1
}
mkdir -p "$parts_dir"
remaining=$(( expected_size - base_size ))
chunk=$(( (remaining + 3) / 4 ))
echo "Recovering $remaining bytes after existing prefix size $base_size in 4 ranges"

pids=()
parts=()
for i in 0 1 2 3; do
  start=$(( base_size + i * chunk ))
  (( start < expected_size )) || break
  end=$(( start + chunk - 1 ))
  (( end < expected_size )) || end=$(( expected_size - 1 ))
  part="$parts_dir/$shard.$start-$end.part"
  wanted=$(( end - start + 1 ))
  parts+=("$part")
  have=0
  if [[ -f "$part" ]]; then
    have=$(stat -c %s "$part")
  fi
  (( have <= wanted )) || { echo "Oversized range file: $part" >&2; exit 1; }
  if (( have == wanted )); then
    echo "Reusing range $start-$end"
    continue
  fi
  resume_start=$(( start + have ))
  echo "Downloading range $resume_start-$end ($have bytes retained)"
  (
    tail_file="$part.tail"
    curl --http1.1 --fail --location --silent --show-error --retry 3 \
      --retry-all-errors --retry-delay 3 --range "$resume_start-$end" \
      --output "$tail_file" "$MODEL_CDN_URL"
    [[ $(stat -c %s "$tail_file") -eq $(( wanted - have )) ]] || exit 1
    cat "$tail_file" >> "$part"
    rm -f "$tail_file"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
(( failed == 0 )) || { echo "A range failed; keep parts for retry" >&2; exit 1; }

for part in "${parts[@]}"; do
  range=${part##*.safetensors.}
  range=${range%.part}
  start=${range%-*}
  end=${range#*-}
  wanted=$(( end - start + 1 ))
  [[ $(stat -c %s "$part") -eq "$wanted" ]] || {
    echo "Wrong size for $part" >&2
    exit 1
  }
done

merged="$parts_dir/$shard.merged"
cp "$partial" "$merged"
for part in "${parts[@]}"; do
  cat "$part" >> "$merged"
done
[[ $(stat -c %s "$merged") -eq "$expected_size" ]] || {
  echo "Merged shard has wrong size" >&2
  exit 1
}
actual_sha=$(sha256sum "$merged" | cut -d ' ' -f 1)
[[ "$actual_sha" == "$expected_sha" ]] || {
  echo "SHA-256 mismatch; preserving partial and parts, not publishing shard" >&2
  exit 1
}
mv "$merged" "$final"
echo "Verified official shard SHA-256: $actual_sha"
echo "Final model file: $final"
