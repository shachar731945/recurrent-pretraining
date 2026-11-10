#!/usr/bin/env bash
# Installs a user-local, full TeX Live distribution for every project under
# /home/shaharav/vinkerg/shahar_avni_folder.  No sudo is required.

set -euo pipefail

GROUP_DIR="/home/shaharav/vinkerg/shahar_avni_folder"
YEAR="2026"
TEXLIVE_ROOT="${TEXLIVE_ROOT:-${GROUP_DIR}/texlive}"
TEXDIR="${TEXLIVE_ROOT}/${YEAR}"
WORK_DIR="${TEXLIVE_ROOT}/.installer-${YEAR}"
INSTALLER_URL="https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz"
ARCH="x86_64-linux"

for command in curl tar perl; do
    command -v "$command" >/dev/null || {
        echo "Error: $command is required but was not found on PATH." >&2
        exit 1
    }
done

if [[ ! -w "$GROUP_DIR" ]]; then
    echo "Error: cannot write to $GROUP_DIR." >&2
    exit 1
fi

if [[ -e "$TEXDIR" ]]; then
    echo "Error: $TEXDIR already exists; refusing to overwrite it." >&2
    echo "To install elsewhere, set TEXLIVE_ROOT before running this script." >&2
    exit 1
fi

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

echo "==> Downloading TeX Live network installer"
echo "    Destination: $TEXDIR"
curl --fail --location --continue-at - --progress-bar \
    --output install-tl-unx.tar.gz "$INSTALLER_URL"

echo "==> Extracting installer"
tar -xzf install-tl-unx.tar.gz
INSTALLER_DIR="$(find . -maxdepth 1 -type d -name 'install-tl-*' -print -quit)"
if [[ -z "$INSTALLER_DIR" || ! -f "$INSTALLER_DIR/install-tl" ]]; then
    echo "Error: could not find the extracted TeX Live installer." >&2
    exit 1
fi

cat > profile <<EOF
selected_scheme scheme-full
TEXDIR $TEXDIR
TEXMFCONFIG $TEXLIVE_ROOT/texmf-config
TEXMFHOME $TEXLIVE_ROOT/texmf-home
TEXMFLOCAL $TEXDIR/texmf-local
TEXMFSYSCONFIG $TEXDIR/texmf-config
TEXMFSYSVAR $TEXDIR/texmf-var
TEXMFVAR $TEXLIVE_ROOT/texmf-var
binary_$ARCH 1
instopt_adjustpath 0
instopt_letter 0
instopt_portable 0
instopt_write18_restricted 1
EOF

echo "==> Installing full TeX Live (this can take a while)"
perl "$INSTALLER_DIR/install-tl" --profile="$WORK_DIR/profile"

BIN_DIR="$TEXDIR/bin/$ARCH"
if [[ ! -x "$BIN_DIR/latex" || ! -x "$BIN_DIR/dvisvgm" ]]; then
    echo "Error: install finished but latex or dvisvgm was not found in $BIN_DIR." >&2
    exit 1
fi

echo
echo "TeX Live installed successfully. Add this once to ~/.bashrc:"
echo "export PATH=\"$BIN_DIR:\$PATH\""
echo
echo "For this terminal only, run:"
echo "export PATH=\"$BIN_DIR:\$PATH\""
echo
echo "Then verify with: latex --version; dvisvgm --version"
