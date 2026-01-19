def fibonacci(n):
    #implement me
    a = 0
    b = 1
    for _ in range(n):
        yield a
        a, b = b, a + b

for t in fibonacci(50):
    print(t)