#!/usr/bin/env bash
set -euo pipefail
NEURO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$NEURO_ROOT/.tools/uv" ]]; then
  NEURO_UV="$NEURO_ROOT/.tools/uv"
elif command -v uv >/dev/null 2>&1; then
  NEURO_UV="$(command -v uv)"
else
  if [[ "$(uname -m)" != "x86_64" ]]; then printf 'Neuro Studio requires x86-64 Linux.\n'; exit 1; fi
  if ! command -v unzip >/dev/null 2>&1; then printf 'Use the release ZIP with bundled uv, or install unzip once.\n'; exit 1; fi
  mkdir -p "$NEURO_ROOT/.tools"
  NEURO_TEMP="$(mktemp -d)"
  trap 'rm -rf "$NEURO_TEMP"' EXIT
  curl --proto '=https' --tlsv1.2 -LsSf 'https://files.pythonhosted.org/packages/06/36/0e7943d9415560aa9fdd775d0bb4b9c06b69c543f0647210e5b84776658b/uv-0.7.8-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl' -o "$NEURO_TEMP/uv.zip"
  printf '81ac0bb371979f48d1293f9c1bee691680ea6a724f16880c8f76718f5ff50049  %s\n' "$NEURO_TEMP/uv.zip" | sha256sum --check --status
  unzip -q "$NEURO_TEMP/uv.zip" -d "$NEURO_TEMP/unpack"
  cp "$NEURO_TEMP/unpack/uv-0.7.8.data/scripts/uv" "$NEURO_ROOT/.tools/uv"
  chmod +x "$NEURO_ROOT/.tools/uv"
  NEURO_UV="$NEURO_ROOT/.tools/uv"
fi
export NEUROSTUDIO_UV="$NEURO_UV"
exec "$NEURO_UV" run --no-project --python 3.12 "$NEURO_ROOT/scripts/bootstrap.py" "$@"
