# def naturals(n):
#     return [i for i in range (n)]

# def naturals(n):
#     for i in range(n):
#         yield i

def naturals(n, i = 0):
    if i < n:
        yield i
        naturals(n, i + 1)

for i in naturals(100):
    print(2 * i)