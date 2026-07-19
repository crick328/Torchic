import timeit

setup = """
class Oggetto:
    def __init__(self, *args):
        self.ctx = args

class OggettoV2:
    def __init__(self, args):
        self.ctx = args
        
def func_args(*args):
    o = Oggetto(*args)

def func_tuple(args):
    o = OggettoV2(args)

a = 1
b = 'a'
c = 3.5
d = "AAA"
"""

t_args = timeit.timeit("func_args(a, b, c, d)", setup=setup, number=10_000_000)
t_tuple = timeit.timeit("func_tuple((a, b, c, d))", setup=setup, number=10000000)
#t_tuple_pre = timeit.timeit("func_tuple(t)", setup=setup, number=10000000)

print(f"Args (variables): {t_args:.5f}s")
print(f"Tuple (created with variables): {t_tuple:.5f}s")
#print(f"Tuple (pre-existing): {t_tuple_pre:.5f}s")