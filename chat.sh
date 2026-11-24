#!/bin/bash
# Chat with Kenchi on a Mac with Apple Silicon (MLX).  The weights folder kenchi-hf sits next to this
# repository; set KENCHI_HF to use another place.  Extra arguments go to run_mlx.py.
cd "$(dirname "$0")"
HF="${KENCHI_HF:-../kenchi-hf}"
exec python run_mlx.py --weights "$HF/model.safetensors" --config "$HF/config.json" "$@"
