from tools import calculator

tests = [
    "0.15 * 2480 + 37 ** 2",                   # expect 1741.0
    "(15.6 * 9/5) + 32",                       # expect about 60.08
    "37 ^ 2",                                  # ^ is XOR in Python: must be rejected
    "__import__('os').system('echo hacked')",  # must be rejected, never executed
    "2 ** 1000",                               # exponent cap: rejected
    "1 / 0",                                   # clean error
]
for t in tests:
    print(t, "->", calculator.invoke({"expression": t}))