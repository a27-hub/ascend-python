CDIR="$(cd "$(dirname "$0")" ; pwd -P)"

cd $CDIR

PYTORCH_VERSION="2.7"
IFS='.' read -ra version_parts <<< "$PYTORCH_VERSION"
PYTORCH_VERSION_DIR="v${version_parts[0]}r${version_parts[1]}"

export PYTORCH_VERSION="$PYTORCH_VERSION"

python3 -m torchnpugen.gen_op_plugin_functions \
  --version="$PYTORCH_VERSION" \
  --output_dir="$CDIR"            \
  --source_yaml="$CDIR/struct_op_functions.yaml" \

python3 -m torchnpugen.gen_derivatives \
  --version="$PYTORCH_VERSION" \
  --output_dir="$CDIR"            \
  --source_yaml="$CDIR/struct_derivatives.yaml" \

python3 -m torchnpugen.struct.gen_struct_opapi \
  --output_dir="$CDIR"            \
  --native_yaml="$CDIR/struct_op_functions.yaml" \
  --struct_yaml="$CDIR/struct_op_functions.yaml" \