#!/usr/bin/env python3
import os
import sys
import time
import subprocess
import json
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests

IMAGE_NAME = "foundry-pure"
MODELS_DIR = "/tmp/foundry-models"
PORT = 8082

CONFIGS = [
    {
        "name": "baseline_q4_k_m",
        "quant": "Qwen3.5-35B-A3B-Q4_K_M.gguf",
        "env": {
            "PROFILE_FLASH_ATTN": "on",
            "PROFILE_KV_TYPE_K": "q8_0",
            "PROFILE_KV_TYPE_V": "q8_0",
            "PROFILE_PARALLEL": "1",
            "PROFILE_EXTRA_ARGS": "--mlock"
        }
    },
    {
        "name": "ud_q4_k_xl_baseline",
        "quant": "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        "env": {
            "PROFILE_FLASH_ATTN": "on",
            "PROFILE_KV_TYPE_K": "q8_0",
            "PROFILE_KV_TYPE_V": "q8_0",
            "PROFILE_PARALLEL": "1",
            "PROFILE_EXTRA_ARGS": "--mlock"
        }
    },
    {
        "name": "ud_q4_k_xl_large_batch",
        "quant": "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        "env": {
            "PROFILE_FLASH_ATTN": "on",
            "PROFILE_KV_TYPE_K": "q8_0",
            "PROFILE_KV_TYPE_V": "q8_0",
            "PROFILE_PARALLEL": "1",
            "PROFILE_EXTRA_ARGS": "--mlock -b 4096 -ub 4096"
        }
    },
    {
        "name": "ud_q4_k_xl_q4_kv_large_batch",
        "quant": "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        "env": {
            "PROFILE_FLASH_ATTN": "on",
            "PROFILE_KV_TYPE_K": "q4_0",
            "PROFILE_KV_TYPE_V": "q4_0",
            "PROFILE_PARALLEL": "1",
            "PROFILE_EXTRA_ARGS": "--mlock -b 4096 -ub 4096"
        }
    },
    {
        "name": "ud_q4_k_xl_no_fa_large_batch",
        "quant": "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        "env": {
            "PROFILE_FLASH_ATTN": "off",
            "PROFILE_KV_TYPE_K": "q8_0",
            "PROFILE_KV_TYPE_V": "q8_0",
            "PROFILE_PARALLEL": "1",
            "PROFILE_EXTRA_ARGS": "--mlock -b 4096 -ub 4096"
        }
    },
]

def run_cmd(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)

def wait_for_server():
    print("Waiting for server to be ready...", end="", flush=True)
    for _ in range(120): # up to 10 minutes
        try:
            resp = requests.get(f"http://localhost:{PORT}/health", timeout=2)
            if resp.status_code == 200:
                print(" Ready!")
                return True
        except:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    print(" Timeout!")
    return False

def run_benchmark(mode="generation"):
    cmd = f"python3 /tmp/foundry-repo/scripts/benchmark.py --url http://localhost:{PORT} --mode {mode} --requests 3"
    result = run_cmd(cmd)
    
    lines = result.stdout.split('\n')
    avg_tps = 0
    prompt_tps = 0
    
    for line in lines:
        if "Average:" in line and "tok/s" in line:
            parts = line.split()
            avg_tps = float(parts[1])
        if "Prompt size ~1000 tokens" in line:
            try:
                time_s = float(line.split()[-1].replace('s', ''))
                prompt_tps = 1000 / time_s
            except:
                pass
                
    return avg_tps, prompt_tps

def main():
    print(f"Starting RTX 5090 Auto-Tuning at {datetime.now().isoformat()}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    results = {}
    
    for conf in CONFIGS:
        print(f"\n{'='*60}")
        print(f"TESTING CONFIG: {conf['name']}")
        print(f"{'='*60}")
        
        run_cmd("docker rm -f foundry-tune 2>/dev/null")
        
        cmd = f"docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all -d --name foundry-tune -p {PORT}:8080 -v {MODELS_DIR}:/models "
        
        cmd += f"-e FOUNDRY_GGUF_FILE={conf['quant']} "
        for k, v in conf['env'].items():
            cmd += f"-e {k}=\"{v}\" "
            
        cmd += f"{IMAGE_NAME}"
        
        print(f"Starting container...")
        run_cmd(cmd)
        
        if not wait_for_server():
            print("Server failed to start. Logs:")
            print(run_cmd("docker logs --tail 50 foundry-tune").stdout)
            continue
            
        print("\nRunning Decoding Benchmark (3 requests)...")
        dec_tps, _ = run_benchmark("generation")
        
        print("\nRunning Encoding Benchmark (Prompt Processing)...")
        _, enc_tps = run_benchmark("prompt")
        
        results[conf['name']] = {
            "decode_tps": dec_tps,
            "encode_tps": enc_tps,
            "quant": conf['quant'],
            "env": conf['env']
        }
        
        print(f"\nRESULTS FOR {conf['name']}:")
        print(f"  Decode: {dec_tps:.2f} tok/s")
        print(f"  Encode: {enc_tps:.2f} tok/s")
        
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    for name, res in results.items():
        print(f"{name:30} | Decode: {res['decode_tps']:6.2f} tok/s | Encode: {res['encode_tps']:8.2f} tok/s")
        
    run_cmd("docker rm -f foundry-tune 2>/dev/null")

if __name__ == "__main__":
    main()
