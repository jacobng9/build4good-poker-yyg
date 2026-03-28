"""Benchmark pkrbot evaluation speed."""
import pkrbot
import time

start = time.perf_counter()
count = 0
for _ in range(1000):
    d = pkrbot.Deck()
    d.shuffle()
    h = d.deal(2)
    b = d.deal(5)
    pkrbot.evaluate(h + b)
    count += 1
elapsed = time.perf_counter() - start
print(f"1000 eval+deal cycles: {elapsed:.4f}s ({count/elapsed:.0f}/sec)")

# Benchmark just evaluate
cards = [pkrbot.Card('As'), pkrbot.Card('Kd'), pkrbot.Card('Qh'), pkrbot.Card('Jc'), pkrbot.Card('Ts'), pkrbot.Card('2d'), pkrbot.Card('3h')]
start = time.perf_counter()
for _ in range(10000):
    pkrbot.evaluate(cards)
elapsed = time.perf_counter() - start
print(f"10000 pure evaluations: {elapsed:.4f}s ({10000/elapsed:.0f}/sec)")

# Benchmark Card creation
start = time.perf_counter()
for _ in range(10000):
    c = pkrbot.Card('As')
elapsed = time.perf_counter() - start
print(f"10000 Card creations: {elapsed:.4f}s ({10000/elapsed:.0f}/sec)")
