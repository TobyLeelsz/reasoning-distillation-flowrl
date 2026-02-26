python - <<'PY'
import os, wandb
print("WANDB_DISABLE_SERVICE =", os.getenv("WANDB_DISABLE_SERVICE"))
run = wandb.init(project="docker-auth-test", name="ping-nosvc", reinit=True)
wandb.log({"ok": 2})
run.finish()
print("wandb init/log OK (no service)")
PY