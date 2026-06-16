"""Quick count of Stage 1 survivors to calibrate filter aggressiveness."""
import time
from src.config import FULL_CANDIDATES_PATH
from src.stage1_filter import stream_and_filter_candidates

start = time.time()
count = 0
for _ in stream_and_filter_candidates(FULL_CANDIDATES_PATH):
    count += 1
    if count % 1000 == 0:
        print(f"  ... {count} survivors so far ({time.time()-start:.1f}s)")

elapsed = time.time() - start
print(f"\nStage 1 retained {count} candidates in {elapsed:.1f}s")
print(f"Estimated Stage 2 embedding time: {count * 0.005:.0f}s "
      f"(at ~5ms/candidate on CPU)")
