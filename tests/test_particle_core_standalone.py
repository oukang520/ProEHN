"""Standard-library known-truth checks; no cohort, ML runtime or model training."""
import importlib.util
import pathlib
import sys
import unittest
import math

path=pathlib.Path(__file__).resolve().parents[1]/'proehn'/'particle_posterior.py'
spec=importlib.util.spec_from_file_location('proehn_particle_core_test',path)
core=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=core
spec.loader.exec_module(core)


class ParticleCoreTests(unittest.TestCase):
    def estimate(self,budget,seed):
        theta=[[0.,0.],[0.,0.]]
        p=core.sample_observation_posterior(theta,[0.,0.],[0.,0.],[0],None,4,particles=budget,random_seed=seed)
        return p,core.posterior_transition_moments(theta,p)[0]

    def test_known_truth_and_unknown_seed(self):
        p,mean=self.estimate(6000,42)
        # Analytic PT=0 posterior: seed0 2/3, seed1 MT0 2/9, seed1 MT1 1/9.
        self.assertEqual({s[-1] for s in p.states},{0,1})
        for key,truth in [(('primary',0),5/9),(('metastasis',0),1/9),(('primary','Metastatic seeding'),1/3)]:
            self.assertLess(abs(mean[key]-truth),.025)
        self.assertAlmostEqual(sum(mean.values()),1.)
        self.assertEqual(self.estimate(50,7),self.estimate(50,7))

    def test_expected_error_improves_with_budget(self):
        errors=[]
        for budget in (32,2048):
            errors.append(sum((self.estimate(budget,seed)[1][('primary',0)]-5/9)**2 for seed in range(8))/8)
        self.assertLess(errors[1],errors[0]/3)

    def test_twelve_genes_use_only_particle_sized_storage(self):
        n=12;theta=[[0.]*(n+1) for _ in range(n+1)]
        p=core.sample_observation_posterior(theta,[0.]*(n+1),[0.]*(n+1),[0]*n,None,4,particles=96,random_seed=1)
        self.assertEqual(len(p.states),96)
        self.assertEqual(len(p.states[0]),25)
        self.assertAlmostEqual(sum(p.weights),1.)

    def test_unsupported_history_is_not_filled_with_seed_plugin(self):
        with self.assertRaises(ValueError):
            core.sample_observation_posterior([[0.,0.],[0.,0.]],[0.,0.],[0.,0.],[1],[0],3,diagnosis_order=1,first_seeding=0,particles=10)


def solve_resolvent(q, diagnosis, start):
    n=len(start)
    matrix=[[(diagnosis[i] if i==j else 0.)-q[i][j] for j in range(n)]+[start[i]] for i in range(n)]
    for j in range(n):
        pivot=max(range(j,n),key=lambda i:abs(matrix[i][j]))
        matrix[j],matrix[pivot]=matrix[pivot],matrix[j]
        value=matrix[j][j]
        matrix[j]=[v/value for v in matrix[j]]
        for i in range(n):
            if i!=j:
                value=matrix[i][j]
                matrix[i]=[x-value*y for x,y in zip(matrix[i],matrix[j])]
    return [row[-1] for row in matrix]


def one_gene_exact(theta,dp,dm,kind,pt,mt,order):
    states=[((i>>0)&1,(i>>1)&1,(i>>2)&1) for i in range(8)]
    q=[[0.]*8 for _ in range(8)]; d_p=[];d_m=[]
    for i,(p,m,seed) in enumerate(states):
        transitions=[]
        if not seed:
            if not p and not m: transitions.append((i+3,math.exp(theta[0][0])))
            transitions.append((i+4,math.exp(theta[1][1]+theta[1][0]*p)))
        else:
            if not p: transitions.append((i+1,math.exp(theta[0][0])))
            if not m: transitions.append((i+2,math.exp(theta[0][0]+theta[0][1])))
        for j,rate in transitions:
            q[j][i]+=rate;q[i][i]-=rate
        d_p.append(math.exp(dp[0]*p+dp[1]*seed))
        d_m.append(seed*math.exp(dm[0]*m+dm[1]*seed))
    initial=[1.]+[0.]*7
    if kind in (0,1,4):
        occupation=solve_resolvent(q,d_p,initial)
        mass=[o*d*(p==pt[0])*(kind==4 or seed==kind) for o,d,(p,m,seed) in zip(occupation,d_p,states)]
    elif kind==2:
        occupation=solve_resolvent(q,[a*(1-state[-1])+b for a,b,state in zip(d_p,d_m,states)],initial)
        mass=[o*d*(m==mt[0]) for o,d,(p,m,seed) in zip(occupation,d_m,states)]
    else:
        first=solve_resolvent(q,[a+b for a,b in zip(d_p,d_m)],initial)
        a=solve_resolvent(q,d_m,[o*d*(p==pt[0]) for o,d,(p,m,seed) in zip(first,d_p,states)])
        b=solve_resolvent(q,d_p,[o*d*(m==mt[0]) for o,d,(p,m,seed) in zip(first,d_m,states)])
        a=[o*d*(m==mt[0]) for o,d,(p,m,seed) in zip(a,d_m,states)]
        b=[o*d*(p==pt[0]) for o,d,(p,m,seed) in zip(b,d_p,states)]
        mass=[x+y for x,y in zip(a,b)] if order==0 else a if order==1 else b
    total=sum(mass);mean={}
    for state,weight in zip(states,mass):
        for key,p in core.transition_probabilities(theta,state).items():
            mean[key]=mean.get(key,0.)+weight/total*p
    return mean


class DenseOracleTests(unittest.TestCase):
    def test_all_observation_processes_against_dense_exact_oracle(self):
        theta=[[.2,-.3],[.4,-.1]];dp=[-.2,.1];dm=[.3,.2]
        cases=[(0,[0],None,None),(1,[0],None,None),(4,[0],None,None),(2,None,[1],None)]
        cases += [(3,[0],[1],order) for order in (0,1,2)]
        for kind,pt,mt,order in cases:
            with self.subTest(kind=kind,order=order):
                truth=one_gene_exact(theta,dp,dm,kind,pt,mt,order)
                samples=core.sample_observation_posterior(theta,dp,dm,pt,mt,kind,diagnosis_order=order,particles=6000,random_seed=17)
                estimate=core.posterior_transition_moments(theta,samples)[0]
                self.assertLess(max(abs(truth.get(k,0)-estimate.get(k,0)) for k in set(truth)|set(estimate)),.04)


if __name__=='__main__': unittest.main()
