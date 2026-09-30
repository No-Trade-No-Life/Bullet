#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
destination="$root/target/native/xgboost-3.2.0"
mkdir -p "$destination"
case "$(uname -s)-$(uname -m)" in
  Darwin-arm64)
    url='https://files.pythonhosted.org/packages/93/f1/c09ef1add609453aa3ba5bafcd0d1c1a805c1263c0b60138ec968f8ec296/xgboost-3.2.0-py3-none-macosx_12_0_arm64.whl'
    sha='eabbd40d474b8dbf6cb3536325f9150b9e6f0db32d18de9914fb3227d0bef5b7'
    library=libxgboost.dylib ;;
  Linux-x86_64)
    url='https://files.pythonhosted.org/packages/ba/1d/c05e982a2220e1a208f5ca8792265cefb43e860e8796bff3245cc94fc2ce/xgboost_cpu-3.2.0-py3-none-manylinux_2_28_x86_64.whl'
    sha='9c10b1653e1a689bf4094a328144cf3633c9be0e5217a76fbdc2fde6abcb2950'
    library=libxgboost.so ;;
  Linux-aarch64)
    url='https://files.pythonhosted.org/packages/16/3d/b996f80a52e6a4f7c5ed1851bfc1f4e8dae2dc7670ef8b0ca283b3006e3b/xgboost_cpu-3.2.0-py3-none-manylinux_2_28_aarch64.whl'
    sha='896b9e2bc52ec0921edf92ddd1cebedf47032e94992b7aca0293524d380ce6e9'
    library=libxgboost.so ;;
  *) echo 'Supported native targets: macOS ARM64, Linux x86_64/aarch64 (glibc >= 2.28)' >&2; exit 1 ;;
esac
archive="$destination/distribution.whl"
if [ ! -f "$archive" ]; then
  temporary=$(mktemp "$destination/download.XXXXXX")
  trap 'rm -f "$temporary"' EXIT HUP INT TERM
  curl --fail --location --retry 2 --max-time 120 "$url" -o "$temporary"
  printf '%s  %s\n' "$sha" "$temporary" | shasum -a 256 --check
  mv "$temporary" "$archive"
fi
printf '%s  %s\n' "$sha" "$archive" | shasum -a 256 --check
# Extract only native code and licenses: no pip install, Python source, or interpreter.
unzip -oq "$archive" 'xgboost/lib/*' -d "$destination"
if [ ! -f "$destination/LICENSE.xgboost" ]; then
  curl --fail --location --retry 2 --max-time 30 \
    https://raw.githubusercontent.com/dmlc/xgboost/v3.2.0/LICENSE -o "$destination/LICENSE.xgboost"
fi
printf '%s  %s\n' b96f9cead5f4ea4c0a217bd9d879427443a353598fb72694f2d8c4428b53a6af "$destination/LICENSE.xgboost" | shasum -a 256 --check
if unzip -Z1 "$archive" | grep -q '^xgboost_cpu.libs/'; then
  unzip -oq "$archive" 'xgboost_cpu.libs/*' -d "$destination"
fi
if [ "$library" = libxgboost.dylib ]; then
  omp="$(brew --prefix libomp)/lib/libomp.dylib"
  test -f "$omp" || { echo 'Install the native libomp package with brew install libomp' >&2; exit 1; }
  # The wrapper also links Homebrew libomp. Both edges must load the same image.
  rm -f "$destination/xgboost/lib/libomp.dylib"
  install_name_tool -id "$destination/xgboost/lib/libxgboost.dylib" "$destination/xgboost/lib/libxgboost.dylib"
  install_name_tool -change @rpath/libomp.dylib "$omp" "$destination/xgboost/lib/libxgboost.dylib"
  codesign --force --sign - "$destination/xgboost/lib/libxgboost.dylib"
fi
# Cargo build configuration only. Applications do not read environment variables.
cat > "$destination/cargo.toml" <<CONFIG
[env]
XGBOOST_LIB_DIR = { value = "$destination/xgboost/lib", force = true }
[build]
rustflags = ["-Lnative=$destination/xgboost/lib", "-Clink-arg=-Wl,-rpath,$destination/xgboost/lib"]
CONFIG
printf '\nNative XGBoost ready. Cargo config: %s/cargo.toml\n' "$destination"

{
  printf 'xgboost=3.2.0\nxgb=3.0.6\narchive_sha256=%s\n' "$sha"
  uname -sm
  shasum -a 256 "$destination/xgboost/lib/$library"
  if [ "$library" = libxgboost.dylib ]; then
    otool -L "$destination/xgboost/lib/$library"
  else
    ldd "$destination/xgboost/lib/$library"
  fi
} > "$destination/native-provenance.txt"
