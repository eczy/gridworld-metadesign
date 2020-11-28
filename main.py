import pandas as pd
import argparse
from collections import defaultdict
import numpy as np


class MDP:
    def __init__(self, csv):
        S = set()
        A = set()
        P = defaultdict(lambda: [])
        R = {}

        self.gamma, self.H = csv[:2, 0]
        for s, a, sprime, p, r in csv[1:, :]:
            S.add(int(s))
            S.add(int(sprime))
            A.add(int(a))
            P[(s, a)].append((int(sprime), p))
            R[(s, a, sprime)] = r

        self.S = list(S)
        self.A = list(A)
        self.P = P
        self.R = R


def value_iteration(mdp, theta=1e-4):
    V = np.zeros(len(mdp.S))
    while True:
        delta = 0.0
        for s in mdp.S:
            v = V[s]
            action_values = np.zeros(len(mdp.A))
            for a in mdp.A:
                for sprime, p in mdp.P[(s, a)]:
                    action_values[a] += p * (
                        mdp.R[(s, a, sprime)] + mdp.gamma * V[sprime]
                    )
            V[s] = np.max(action_values)
            delta = np.max([delta, np.abs(v - V[s])])
        if delta < theta:
            break
    policy = {}
    for s in mdp.S:
        action_values = np.zeros(len(mdp.A))
        for a in mdp.A:
            for sprime, p in mdp.P[(s, a)]:
                action_values[a] += p * (mdp.R[(s, a, sprime)] + mdp.gamma * V[sprime])
        policy[s] = np.argmax(action_values)
    return V, policy


def policy_evaluation(mdp, pi, theta=1e-4):
    V = np.zeros(len(mdp.S))
    while True:
        delta = 0
        for s in mdp.S:
            v = V[s]
            V[s] = 0
            for sprime, p in mdp.P[(s, pi[s])]:
                V[s] += p * (mdp.R[(s, pi[s], sprime)] + mdp.gamma * V[sprime])
            delta = np.max([delta, np.abs(v - V[s])])
        if delta < theta:
            break
    return V


def policy_improvement(mdp, V, pi):
    for s in mdp.S:
        b = pi[s]
        action_values = np.zeros(len(mdp.A))
        for a in mdp.A:
            for sprime, p in mdp.P[(s, a)]:
                action_values[a] += p * (mdp.R[(s, a, sprime)] + mdp.gamma * V[sprime])
        pi[s] = np.argmax(action_values)
        if b != pi[s]:
            return False
    return True


def policy_iteration(mdp):
    V = np.zeros(len(mdp.S))
    pi = {s: np.random.choice(mdp.A) for s in mdp.S}
    while True:
        V = policy_evaluation(mdp, pi)
        if policy_improvement(mdp, V, pi):
            break
    return V, pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()

    mdp = MDP(pd.read_csv(args.path, index_col=None, header=None).values)

    print(mdp.S)
    V, pi = value_iteration(mdp)
    print(V)
    print(pi)
    V, pi = policy_iteration(mdp)
    print(V)
    print(pi)


if __name__ == "__main__":
    main()
