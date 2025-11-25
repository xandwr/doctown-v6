"""
Benchmark script to compare original vs optimized local LLM performance.

Usage:
    python -m app.llm.benchmark_local --model Qwen/Qwen2.5-Coder-7B-Instruct --chunks 50 --batch-size 8
"""
import argparse
import asyncio
import time
from typing import List

from ..ingestors.base import UniversalChunk, Domain, ChunkType, ChunkMetadata
from .local_client import create_local_generator
from .local_client_optimized import create_optimized_local_generator


def create_sample_chunks(n: int = 50) -> List[UniversalChunk]:
    """Create sample chunks for benchmarking."""
    chunks = []
    
    sample_functions = [
        '''def calculate_total(items: List[Item]) -> float:
    """Calculate total price with tax."""
    subtotal = sum(item.price for item in items)
    tax = subtotal * 0.08
    return subtotal + tax''',
        
        '''class DataProcessor:
    """Process and validate data."""
    def __init__(self, config: Config):
        self.config = config
        self.cache = {}
    
    def process(self, data: dict) -> dict:
        if data["id"] in self.cache:
            return self.cache[data["id"]]
        result = self._transform(data)
        self.cache[data["id"]] = result
        return result''',
        
        '''async def fetch_user_data(user_id: int) -> UserData:
    """Fetch user data from API."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"/api/users/{user_id}") as resp:
            data = await resp.json()
            return UserData.from_dict(data)''',
    ]
    
    for i in range(n):
        code = sample_functions[i % len(sample_functions)]
        chunk = UniversalChunk(
            chunk_id=f"chunk_{i}",
            text=code,
            domain=Domain.CODE,
            type=ChunkType.FUNCTION,
            path=f"src/module_{i}.py",
            metadata=ChunkMetadata(
                line_start=1,
                line_end=10,
            ),
        )
        chunks.append(chunk)
    
    return chunks


async def benchmark_original(chunks: List[UniversalChunk], model_id: str):
    """Benchmark original LocalDocGenerator."""
    print(f"\n{'='*60}")
    print("ORIGINAL LocalDocGenerator")
    print(f"{'='*60}")
    
    generator = create_local_generator(
        model_id=model_id,
        quantization="4bit",
    )
    
    # Load model (not counted in benchmark time)
    print("Loading model...")
    generator.model.load_model()
    print("✓ Model loaded\n")
    
    # Benchmark
    print(f"Processing {len(chunks)} chunks sequentially...")
    start_time = time.time()
    
    result = await generator.generate_batch_async(
        chunks=chunks,
        max_concurrent=1,  # Sequential
        progress_callback=lambda done, total: print(f"  Progress: {done}/{total}", end="\r"),
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Time per chunk: {elapsed/len(chunks):.3f}s")
    print(f"Successful: {result.successful}/{len(chunks)}")
    print(f"Total tokens: {result.total_tokens:,}")
    
    return elapsed


async def benchmark_optimized(
    chunks: List[UniversalChunk],
    model_id: str,
    batch_size: int = 8,
    use_speculative: bool = False,
):
    """Benchmark optimized OptimizedLocalDocGenerator."""
    print(f"\n{'='*60}")
    print("OPTIMIZED OptimizedLocalDocGenerator")
    print(f"{'='*60}")
    
    generator = create_optimized_local_generator(
        model_id=model_id,
        quantization="4bit",
        max_batch_size=batch_size,
        enable_speculative_decoding=use_speculative,
    )
    
    # Load model (not counted in benchmark time)
    print("Loading model...")
    generator.model.load_model()
    print("✓ Model loaded")
    
    # Precompute system prompt cache
    print("Pre-computing system prompt cache...")
    generator._ensure_system_prompt_cached(Domain.CODE)
    print("✓ System prompt cached\n")
    
    # Benchmark
    print(f"Processing {len(chunks)} chunks in batches of {batch_size}...")
    start_time = time.time()
    
    result = await generator.generate_batch_async(
        chunks=chunks,
        max_concurrent=batch_size,
        progress_callback=lambda done, total: print(f"  Progress: {done}/{total}", end="\r"),
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Time per chunk: {elapsed/len(chunks):.3f}s")
    print(f"Successful: {result.successful}/{len(chunks)}")
    print(f"Total tokens: {result.total_tokens:,}")
    
    return elapsed


async def main():
    parser = argparse.ArgumentParser(description="Benchmark local LLM performance")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=50,
        help="Number of chunks to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for optimized generator",
    )
    parser.add_argument(
        "--skip-original",
        action="store_true",
        help="Skip original generator (faster benchmark)",
    )
    parser.add_argument(
        "--speculative",
        action="store_true",
        help="Enable speculative decoding (requires draft model)",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK: Local LLM Performance")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Chunks: {args.chunks}")
    print(f"Batch size: {args.batch_size}")
    if args.speculative:
        print(f"Speculative decoding: ENABLED")
    print(f"{'='*60}")
    
    # Create sample chunks
    print("\nGenerating sample chunks...")
    chunks = create_sample_chunks(args.chunks)
    print(f"✓ Created {len(chunks)} sample chunks")
    
    # Benchmark original
    original_time = None
    if not args.skip_original:
        try:
            original_time = await benchmark_original(chunks, args.model)
        except Exception as e:
            print(f"\n❌ Original generator failed: {e}")
    
    # Benchmark optimized
    try:
        optimized_time = await benchmark_optimized(
            chunks,
            args.model,
            args.batch_size,
            args.speculative,
        )
    except Exception as e:
        print(f"\n❌ Optimized generator failed: {e}")
        return
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if original_time:
        print(f"Original:  {original_time:7.2f}s  ({original_time/len(chunks):.3f}s per chunk)")
        print(f"Optimized: {optimized_time:7.2f}s  ({optimized_time/len(chunks):.3f}s per chunk)")
        speedup = original_time / optimized_time
        print(f"\n🚀 SPEEDUP: {speedup:.1f}x faster!")
    else:
        print(f"Optimized: {optimized_time:7.2f}s  ({optimized_time/len(chunks):.3f}s per chunk)")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
