import json
import os
import random
import time
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime

import docker
import requests
from huggingface_hub import snapshot_download

# Each entry: (base_model, lora_hf_id, local_lora_path)
# lora_hf_id and local_lora_path are mutually exclusive — set the unused one to None.
model_list = [
    # ("codellama/CodeLlama-7b-Instruct-hf", None, None),
    # ("mistralai/Mistral-7B-Instruct-v0.2", "iamPi/leduc_poker-v1.4.0-test_mistralai_Mistral-7B-Instruct-v0.2", None),
    # ("NousResearch/Hermes-3-Llama-3.2-3B", "iamPi/leduc_poker-v1.4.0-test_NousResearch_Hermes-3-Llama-3.2-3B", None),
    # ("Qwen/Qwen2.5-3B-Instruct", None, None),
    # ("Qwen/Qwen2.5-7B-Instruct", "iamPi/leduc_poker-v1.4.0-test_Qwen_Qwen2.5-7B-Instruct", None),
    # ("Qwen/Qwen2-7B-Instruct", "iamPi/leduc_poker-v1.4.0-test_Qwen_Qwen2-7B-Instruct", None),
    # ("unsloth/Llama-3.2-3B-Instruct", "iamPi/leduc_poker-v1.5.1-test_unsloth_Llama-3.2-3B-Instruct", None),
    # ("mistralai/Mistral-7B-Instruct-v0.3", None, "outputs/1/checkpoint-100"),
]

# --- Model Configuration (can override via env: BASE_MODEL_NAME, LORA_MODEL_NAME, LOCAL_LORA_PATH) ---
BASE_MODEL_NAME = os.environ.get("BASE_MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.3")
BASE_MODEL_REVISION = os.environ.get("BASE_MODEL_REVISION") or None
LORA_MODEL_NAME = os.environ.get("LORA_MODEL_NAME", "iamPi/gin_rummy-v1.1.0-mistral7b0.3") or ""
LORA_MODEL_REVISION = os.environ.get("LORA_MODEL_REVISION") or None
# Local path to a LoRA checkpoint folder. When set, skips HF upload/download entirely.
# Takes precedence over LORA_MODEL_NAME. Use an absolute path or a path relative to this script.
LOCAL_LORA_PATH = os.environ.get("LOCAL_LORA_PATH", "") or ""

# --- Evaluation Configuration ---
# All settings can be overridden via environment variables (e.g. when called automatically after training).
GAME_TO_EVAL = os.environ.get("GAME_TO_EVAL", "gin_rummy")  # Options: goofspiel, liars_dice, leduc_poker, gin_rummy, othello, backgammon, hex, clobber
OPPONENT_TYPE = os.environ.get("OPPONENT_TYPE", "mcts")
MCTS_MAX_SIMULATIONS = int(os.environ.get("MCTS_MAX_SIMULATIONS", "25"))
MCTS_NUM_ROLLOUTS = int(os.environ.get("MCTS_NUM_ROLLOUTS", "1"))
NUM_EVALS = int(os.environ.get("NUM_EVALS", "200"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.0"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
NUM_CONCURRENT_EVAL_WORKERS = int(os.environ.get("NUM_CONCURRENT_EVAL_WORKERS", "10"))
NUM_AGENTGYM_SERVERS = int(os.environ.get("NUM_AGENTGYM_SERVERS", "10"))  # Number of parallel agentgym server instances
REUSE_AGENTGYM_SERVERS = os.environ.get("REUSE_AGENTGYM_SERVERS", "0") == "1"

# Number of GPUs for SGLang (env: NUM_GPUS). Uses tensor parallelism when > 1.
NUM_GPUS = int(os.environ.get("NUM_GPUS", "2"))

##############################################################################################

client = docker.from_env()

GAMES_TO_TASK_ID_RANGE = {
    "goofspiel": (0, 99999999),
    "liars_dice": (100000000, 199999999),
    "leduc_poker": (200000000, 299999999),
    "gin_rummy": (300000000, 399999999),
    "othello": (400000000, 499999999),
    "backgammon": (500000000, 599999999),
    "hex": (600000000, 699999999),
    "clobber": (700000000, 799999999),
}
SGLANG_IMAGE = "lmsysorg/sglang:latest"
AGENTGYM_IMAGE = "phoenixbeaudry/game:mcts-api"
NETWORK_NAME = "agent_eval_net"
SGLANG_PORT = 30000
HF_CACHE_DIR = "/mnt/hf_cache"
task_id_range = GAMES_TO_TASK_ID_RANGE[GAME_TO_EVAL]
task_id_min, task_id_max = task_id_range
DATA_LEN_RANGE = task_id_max

def _cache_path_to_hf_id(cache_path: str) -> str:
    """Convert a Docker training cache path to a HuggingFace model ID.

    e.g. /cache/models/mistralai--Mistral-7B-Instruct-v0.3
      -> mistralai/Mistral-7B-Instruct-v0.3
    """
    name = os.path.basename(cache_path.rstrip("/"))
    return name.replace("--", "/", 1)


def _fix_local_lora_paths(lora_dir: str, expected_base_model: str) -> None:
    """Patch adapter_config.json and README.md in a local LoRA checkpoint so that
    the base_model field contains the HF model ID instead of the Docker cache path
    written during training (e.g. /cache/models/org--model -> org/model).
    Edits are made in-place and are idempotent.
    """
    # --- adapter_config.json ---
    adapter_cfg_path = os.path.join(lora_dir, "adapter_config.json")
    if os.path.isfile(adapter_cfg_path):
        with open(adapter_cfg_path) as f:
            cfg = json.load(f)
        raw = cfg.get("base_model_name_or_path", "")
        if raw and raw != expected_base_model:
            converted = _cache_path_to_hf_id(raw)
            if converted != raw:
                cfg["base_model_name_or_path"] = converted
                with open(adapter_cfg_path, "w") as f:
                    json.dump(cfg, f, indent=2)
                print(f"  ✓ Patched adapter_config.json: {raw} -> {converted}")

    # --- Stale tokenizer files that break SGLang LoRA loading ---
    for fname in ("merges.txt", "added_tokens.json", "vocab.json"):
        fpath = os.path.join(lora_dir, fname)
        if os.path.isfile(fpath):
            os.remove(fpath)
            print(f"  ✓ Removed {fname} (not needed for LoRA serving)")

    # --- README.md (HF model card) ---
    readme_path = os.path.join(lora_dir, "README.md")
    if os.path.isfile(readme_path):
        with open(readme_path) as f:
            text = f.read()
        import re
        def _replace_cache_path(m):
            converted = _cache_path_to_hf_id(m.group(1))
            return m.group(0).replace(m.group(1), converted)
        patched = re.sub(r"(/cache/models/[\w\-\.]+)", _replace_cache_path, text)
        if patched != text:
            with open(readme_path, "w") as f:
                f.write(patched)
            print("  ✓ Patched README.md base_model paths")


def run_evaluation():
    start_time = time.time()
    containers = {}
    avg_score = 0.0

    try:
        # 1. Infrastructure Setup
        networks = client.networks.list(names=[NETWORK_NAME])
        if not networks:
            client.networks.create(NETWORK_NAME, driver="bridge")

        lora_dir = None
        if LOCAL_LORA_PATH:
            lora_dir = os.path.abspath(LOCAL_LORA_PATH)
            _fix_local_lora_paths(lora_dir, BASE_MODEL_NAME)
            print(f"🚀 Starting SGLang: {BASE_MODEL_NAME} w/ local lora {lora_dir}")
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
                "--enable-lora --lora-paths trained_lora=/lora/trained_lora "
                "--lora-backend triton "
                f"--host 0.0.0.0 --port 30000 --tensor-parallel-size {NUM_GPUS} --dtype float16 "
                f"--random-seed {RANDOM_SEED}"
            )
        elif LORA_MODEL_NAME:
            print(f"🚀 Starting SGLang: {BASE_MODEL_NAME} w/ lora {LORA_MODEL_NAME}")
            safe_lora_name = LORA_MODEL_NAME.replace("/", "_")
            lora_dir = f"/tmp/sglang_lora/{safe_lora_name}"
            print(f"⬇️  Downloading LoRA to {lora_dir}...")
            snapshot_download(
                repo_id=LORA_MODEL_NAME,
                revision=LORA_MODEL_REVISION,
                local_dir=lora_dir,
                local_dir_use_symlinks=False,
            )
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
                "--enable-lora --lora-paths trained_lora=/lora/trained_lora "
                "--lora-backend triton "
                f"--host 0.0.0.0 --port 30000 --tensor-parallel-size {NUM_GPUS} --dtype float16 "
                f"--random-seed {RANDOM_SEED}"
            )
        else:
            print(f"🚀 Starting SGLang: {BASE_MODEL_NAME}" + (f" (tensor_parallel_size={NUM_GPUS})" if NUM_GPUS > 1 else ""))
            sglang_command = (
                f"python3 -m sglang.launch_server --model-path {BASE_MODEL_NAME} "
                f"{'--revision ' + BASE_MODEL_REVISION if BASE_MODEL_REVISION else ''} "
                f"--host 0.0.0.0 --port 30000 --tensor-parallel-size {NUM_GPUS} --dtype float16 "
                f"--random-seed {RANDOM_SEED}"
            )

        sglang = client.containers.run(
            SGLANG_IMAGE,
            command=sglang_command,
            name="sglang-server",
            detach=True,
            network=NETWORK_NAME,
            ports={f"{SGLANG_PORT}/tcp": SGLANG_PORT},
            device_requests=[docker.types.DeviceRequest(count=NUM_GPUS, capabilities=[['gpu']])],
            environment={
                "HF_HOME": "/hf",
                "TRANSFORMERS_CACHE": "/hf",
                "HUGGINGFACE_HUB_CACHE": "/hf",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "PYTHONHASHSEED": str(RANDOM_SEED),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "NVIDIA_TF32_OVERRIDE": "0",
            },
            volumes={
                HF_CACHE_DIR: {"bind": "/hf", "mode": "rw"},
                **({lora_dir: {"bind": "/lora/trained_lora", "mode": "ro"}} if lora_dir else {}),
            },
            ipc_mode="host",
        )
        containers['sglang'] = sglang

        agentgym_ports = [8001 + i for i in range(NUM_AGENTGYM_SERVERS)]
        if REUSE_AGENTGYM_SERVERS:
            print(f"Reusing {NUM_AGENTGYM_SERVERS} existing AgentGym server(s) on ports {agentgym_ports[0]}–{agentgym_ports[-1]}")
        else:
            print(f"🚀 Starting {NUM_AGENTGYM_SERVERS} AgentGym Server(s)...")
            for i, port in enumerate(agentgym_ports):
                agent = client.containers.run(
                    AGENTGYM_IMAGE,
                    name=f"agentgym-server-{i}",
                    detach=True,
                    network=NETWORK_NAME,
                    ports={'8000/tcp': port}
                )
                containers[f'agent-{i}'] = agent
                print(f"  ✓ AgentGym server {i+1}/{NUM_AGENTGYM_SERVERS} on port {port}")

        # 2. Wait for Readiness
        print("⏳ Waiting for SGLang health check...")
        while True:
            try:
                if requests.get(f"http://localhost:{SGLANG_PORT}/v1/models", timeout=2).status_code == 200:
                    break
            except:
                time.sleep(5)
        print("✅ SGLang Ready.\n")

        # 3. Evaluation Loop
        random.seed(RANDOM_SEED)
        eval_list = random.sample(range(task_id_min + 1, task_id_max), NUM_EVALS)
        total_score = 0.0

        if LOCAL_LORA_PATH or LORA_MODEL_NAME:
            # For OpenAI-compatible API, use base-model:adapter-name format per SGLang docs
            # Format: model_path:adapter_name (e.g., "Qwen/Qwen2.5-3B-Instruct:trained_lora")
            inference_model_name = f"{BASE_MODEL_NAME}:trained_lora"
        else:
            inference_model_name = BASE_MODEL_NAME

        # Distribute tasks across agentgym servers (round-robin)
        task_to_server = {task_id: agentgym_ports[i % NUM_AGENTGYM_SERVERS]
                          for i, task_id in enumerate(eval_list)}

        def evaluate_task(task_id, server_port):
            payload = {
                "model": inference_model_name,
                "base_url": f"http://sglang-server:{SGLANG_PORT}/v1",
                "task_id": task_id,
                "temperature": TEMPERATURE,
                "seed": task_id,
                "opponent": OPPONENT_TYPE,
                "api_key": "test",
                "mcts_max_simulations": MCTS_MAX_SIMULATIONS,
                "mcts_num_rollouts": MCTS_NUM_ROLLOUTS
            }
            try:
                response = requests.post(f"http://localhost:{server_port}/evaluate", json=payload, timeout=2500)
                result = response.json()
                result_payload = result.get("result") if isinstance(result, dict) else None
                if isinstance(result_payload, dict):
                    data = result_payload
                else:
                    data = result if isinstance(result, dict) else {}
                score = data.get('score', 0.0)
                return task_id, score, data, None
            except Exception as e:
                return task_id, 0.0, {}, str(e)

        print(f"Running {NUM_EVALS} evaluations with concurrency={NUM_CONCURRENT_EVAL_WORKERS} "
              f"across {NUM_AGENTGYM_SERVERS} server(s)...")
        completed = 0
        all_results = {}  # Store full results: {task_id: full_result_data}

        with ThreadPoolExecutor(max_workers=NUM_CONCURRENT_EVAL_WORKERS) as executor:
            futures = {executor.submit(evaluate_task, task_id, task_to_server[task_id]): task_id
                      for task_id in eval_list}
            for future in as_completed(futures):
                task_id, score, full_data, error = future.result()
                completed += 1
                total_score += score

                # Store full result
                all_results[task_id] = {
                    "task_id": task_id,
                    "score": score,
                    "error": error,
                    "full_result": full_data
                }

                if error:
                    print(f"[{completed}/{NUM_EVALS}] Task {task_id}: FAILED ({error})")
                else:
                    print(f"[{completed}/{NUM_EVALS}] Task {task_id}: {score}")

        # 4. Save Results to a folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_safe = GAME_TO_EVAL.replace("_", "-")
        model_safe = BASE_MODEL_NAME.replace("/", "_").replace(" ", "_")
        if LORA_MODEL_NAME:
            model_safe = model_safe + "_" + LORA_MODEL_NAME.replace("/", "_").replace(" ", "_")
        eval_results_dir = os.path.join("eval_results", game_safe, model_safe)
        os.makedirs(eval_results_dir, exist_ok=True)

        json_filename = os.path.join(eval_results_dir, f"eval_results_{game_safe}_{model_safe}_{timestamp}.json")
        scores_filename = os.path.join(eval_results_dir, f"eval_scores_{game_safe}_{model_safe}_{timestamp}.txt")

        # Save full results to JSON
        json_output = {
            "config": {
                "game": GAME_TO_EVAL,
                "opponent_type": OPPONENT_TYPE,
                "num_evals": NUM_EVALS,
                "temperature": TEMPERATURE,
                "random_seed": RANDOM_SEED,
                "num_concurrent_workers": NUM_CONCURRENT_EVAL_WORKERS,
                "num_agentgym_servers": NUM_AGENTGYM_SERVERS,
                "base_model": BASE_MODEL_NAME,
                "lora_model": LORA_MODEL_NAME,
            },
            "summary": {
                "total_score": total_score,
                "num_evals": NUM_EVALS,
                "avg_score": total_score / NUM_EVALS if NUM_EVALS > 0 else 0,
            },
            "results": all_results
        }

        with open(json_filename, 'w') as f:
            json.dump(json_output, f, indent=2)
        print(f"💾 Full results saved to: {json_filename}")

        # Save scores to simple text file
        with open(scores_filename, 'w') as f:
            f.write(f"Evaluation Scores - {GAME_TO_EVAL}\n")
            avg = total_score / NUM_EVALS if NUM_EVALS > 0 else 0
            f.write(f"Total: {total_score}/{NUM_EVALS} (Avg: {avg:.4f})\n")
            f.write("-" * 50 + "\n")
            for task_id in sorted(all_results.keys()):
                result = all_results[task_id]
                if result["error"]:
                    f.write(f"Task {task_id}: FAILED ({result['error']})\n")
                else:
                    f.write(f"Task {task_id}: {result['score']}\n")
        print(f"💾 Scores saved to: {scores_filename}")

        # 5. Final Score
        avg_score = total_score / NUM_EVALS if NUM_EVALS > 0 else 0

        print("\n✅ Evaluation complete.")
        print(f"Score: {total_score}/{NUM_EVALS} ({avg_score:.4f})")

    finally:
        print("🧹 Cleaning up containers...")
        for name, c in containers.items():
            if REUSE_AGENTGYM_SERVERS and name.startswith('agent-'):
                continue  # caller owns the agentgym container lifecycle
            try:
                c.remove(force=True)
                print(f"  ✓ Removed {name}")
            except Exception as e:
                print(f"  ⚠ Failed to remove {name}: {e}")

        end_time = time.time()
        print(f"Evaluation completed in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    # When invoked automatically after training (e.g. from e2e.sh), the caller
    # sets BASE_MODEL_NAME (and optionally LOCAL_LORA_PATH / LORA_MODEL_NAME) via env vars so we
    # skip the manual model_list iteration and evaluate just that one model.
    if os.environ.get("BASE_MODEL_NAME"):
        run_evaluation()
    else:
        for base_model_name, lora_model_name, local_lora_path in model_list:
            BASE_MODEL_NAME = base_model_name
            LORA_MODEL_NAME = lora_model_name or ""
            LOCAL_LORA_PATH = local_lora_path or ""
            run_evaluation()