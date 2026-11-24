# bipartite schedule: hosts H0,H1 only host; visitors V0..V3 only visit, each <=2 matches. Rule 0-1-3, home AC slopes xi.
import numpy as np, itertools
xi=np.array([0,1/3,1]); a=np.array([0,.4,.1])
def Ph(z): e=np.exp(a+xi*z); return e/e.sum()
def Pa(z): return Ph(-z)[::-1]          # reciprocity: visitor gets y <=> host gets L-1-y
def dl(P,y,z,h=1e-6): return (np.log(P(z+h)[y])-np.log(P(z-h)[y]))/(2*h)
th={'H0':.5,'H1':-.2,'V0':.3,'V1':-.6,'V2':.1,'V3':.9}
# visitor V0 visits H0 and H1
by={}
for ys in itertools.product(range(3),repeat=2):
    s=round(sum(xi[y] for y in ys),9)
    g=dl(Pa,ys[0],th['V0']-th['H0'])+dl(Pa,ys[1],th['V0']-th['H1'])
    by.setdefault(s,[]).append(g)
print("visitor, 2 matches, spread:",max(np.ptp(v) for v in by.values()))
# host H0 hosts V0..V3 (4 matches)
by={}
for ys in itertools.product(range(3),repeat=4):
    s=round(sum(xi[y] for y in ys),9)
    g=sum(dl(Ph,y,th['H0']-th[v]) for y,v in zip(ys,['V0','V1','V2','V3']))
    by.setdefault(s,[]).append(g)
print("host, 4 matches, spread:",max(np.ptp(v) for v in by.values()))
