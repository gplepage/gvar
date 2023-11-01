from gvar import *
from numpy import *

# build data for format tests

cases = """
3.14156(590)
-3.14156(590)
0.012345678 +- 9.996
314156789 +- 0.590
3.14156789e15 +- 0.590
0.59 +- 31415678901
3.14159654 +- 0
0 +- 3.14159
3.14159654 +- inf
inf +- 3.14159
0 +- 0
""".split('\n')[1:-1]

glist = gvar(cases)
print('cases = {')
print(f"    'inputs':{cases},")
for f in ['{}', '{:.2e}', '{:.2f}', '{:.4g}', '{:.2p}', '{:#.2p}', '{:.2P}', '{:.^20.2P}']:
    print(f"    '{f}':{list(fmt(glist, format=f))},")
print('    }')