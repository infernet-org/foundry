#!/usr/bin/env python3
"""
Foundry Benchmark Script
Measures token generation speed, time-to-first-token, and throughput for Qwen3.5-35B-A3B
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

try:
    import requests
except ImportError:
    print("Installing requests...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "requests"])
    import requests

DEFAULT_MODEL = "qwen3.5-35b-a3b"


def benchmark_generation_speed(
    base_url: str,
    model: str,
    num_requests: int = 5,
    max_tokens: int = 128,
) -> dict:
    """Measure tokens per second during generation."""
    print(f"\n{'=' * 60}")
    print("GENERATION SPEED BENCHMARK")
    print(f"{'=' * 60}")
    print(f"Requests: {num_requests} | Max tokens: {max_tokens}\n")

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Count from 1 to 100, list each number on a new line. Be thorough.",
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }

    results: dict = {
        "ttft_ms": [],
        "tps": [],
        "total_tokens": [],
        "response_time": [],
    }

    for i in range(num_requests):
        try:
            print(f"Request {i + 1}/{num_requests}...", end=" ", flush=True)
            start_time = time.time()

            response = requests.post(url, json=payload, timeout=120)
            total_time = time.time() - start_time

            if response.status_code != 200:
                print(f"ERROR: {response.status_code}")
                continue

            data = response.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            tps = tokens / total_time if total_time > 0 else 0

            results["tps"].append(tps)
            results["total_tokens"].append(tokens)
            results["response_time"].append(total_time)

            print(f"{tokens} tokens in {total_time:.2f}s = {tps:.2f} tok/s")

        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"ERROR: {e}")
            continue

        time.sleep(0.5)

    if results["tps"]:
        avg_tps = sum(results["tps"]) / len(results["tps"])
        min_tps = min(results["tps"])
        max_tps = max(results["tps"])
        print("\nResults:")
        print(f"  Average:    {avg_tps:.2f} tok/s")
        print(f"  Min:        {min_tps:.2f} tok/s")
        print(f"  Max:        {max_tps:.2f} tok/s")
        print(f"  Avg tokens: {sum(results['total_tokens']) / len(results['total_tokens']):.1f}")
        print(f"  Avg time:   {sum(results['response_time']) / len(results['response_time']):.2f}s")

    return results


def benchmark_prompt_processing(
    base_url: str,
    model: str,
    prompt_sizes: list[int] | None = None,
) -> dict:
    """Measure prompt processing speed (tokens/sec for prefill)."""
    if prompt_sizes is None:
        prompt_sizes = [100, 500, 1000]

    print(f"\n{'=' * 60}")
    print("PROMPT PROCESSING SPEED")
    print(f"{'=' * 60}\n")

    url = f"{base_url}/v1/chat/completions"
    results: dict = {}

    for size in prompt_sizes:
        try:
            # Create a prompt of approximately `size` tokens
            prompt = ("The quick brown fox jumps over the lazy dog. " * (size // 10))[: size * 4]

            print(f"Prompt size ~{size} tokens...", end=" ", flush=True)

            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.7,
            }

            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            total_time = time.time() - start_time

            if response.status_code == 200:
                print(f"{total_time:.3f}s")
                results[size] = total_time
            else:
                print(f"ERROR: {response.status_code}")

        except (requests.RequestException, ValueError) as e:
            print(f"ERROR: {e}")

        time.sleep(0.5)

    return results


def benchmark_concurrent_throughput(
    base_url: str,
    model: str,
    num_concurrent: int = 4,
    num_iterations: int = 3,
) -> None:
    """Measure sustained throughput with multiple concurrent requests."""
    print(f"\n{'=' * 60}")
    print("CONCURRENT THROUGHPUT")
    print(f"{'=' * 60}")
    print(f"Concurrent slots: {num_concurrent} | Iterations: {num_iterations}\n")

    import concurrent.futures

    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Write a haiku about artificial intelligence"}],
        "max_tokens": 50,
        "temperature": 0.7,
    }

    def make_request() -> tuple:
        try:
            start = time.time()
            response = requests.post(url, json=payload, timeout=60)
            elapsed = time.time() - start
            if response.status_code == 200:
                tokens = response.json().get("usage", {}).get("completion_tokens", 0)
                return tokens, elapsed
        except (requests.RequestException, ValueError, KeyError) as e:
            print(f"Request failed: {e}")
        return None, None

    total_tokens = 0
    total_time = 0
    request_count = 0

    for iteration in range(num_iterations):
        print(f"Iteration {iteration + 1}/{num_iterations}: ", end="", flush=True)

        start_batch = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(make_request) for _ in range(num_concurrent)]
            batch_tokens = 0
            for future in concurrent.futures.as_completed(futures):
                tokens, _ = future.result()
                if tokens:
                    batch_tokens += tokens
                    total_tokens += tokens
                    request_count += 1

        batch_time = time.time() - start_batch
        total_time += batch_time
        batch_tps = batch_tokens / batch_time if batch_time > 0 else 0

        print(f"{batch_tokens} tokens in {batch_time:.2f}s = {batch_tps:.2f} tok/s")
        time.sleep(0.5)

    if total_time > 0:
        avg_tps = total_tokens / total_time
        print("\nOverall:")
        print(f"  Total tokens:      {total_tokens}")
        print(f"  Total time:        {total_time:.2f}s")
        print(f"  Average throughput: {avg_tps:.2f} tok/s")
        print(f"  Requests completed: {request_count}")


def health_check(base_url: str) -> bool:
    """Check if server is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"[OK] Server is healthy at {base_url}")
            return True
    except requests.RequestException:
        pass

    print(f"[FAIL] Server not responding at {base_url}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Foundry Benchmark")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name for API requests")
    parser.add_argument(
        "--mode",
        choices=["all", "generation", "prompt", "throughput"],
        default="all",
        help="Benchmark mode",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=5,
        help="Number of requests for generation bench",
    )
    parser.add_argument("--concurrent", type=int, default=4, help="Concurrent requests")
    parser.add_argument("--output", help="Save results to JSON file")

    args = parser.parse_args()

    print(f"\n{'=' * 60}")
    print("FOUNDRY BENCHMARK")
    print(f"{'=' * 60}")
    print(f"Server: {args.url}")
    print(f"Time: {datetime.now().isoformat()}\n")

    if not health_check(args.url):
        sys.exit(1)

    results: dict = {}

    if args.mode in ["all", "generation"]:
        results["generation"] = benchmark_generation_speed(
            args.url,
            model=args.model,
            num_requests=args.requests,
        )

    if args.mode in ["all", "prompt"]:
        results["prompt_processing"] = benchmark_prompt_processing(
            args.url,
            model=args.model,
        )

    if args.mode in ["all", "throughput"]:
        benchmark_concurrent_throughput(
            args.url,
            model=args.model,
            num_concurrent=args.concurrent,
        )

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print(f"\n{'=' * 60}\n")


if __name__ == "__main__":
    main()
