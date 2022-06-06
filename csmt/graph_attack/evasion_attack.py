
from csmt.graph_attack.modification.rand import RAND
from csmt.graph_attack.modification.stack import STACK
from csmt.graph_attack.modification.flip import FLIP


def evasion_dict(attack_algorithm,n_edge_mod):
    if attack_algorithm[0]=='random':
        return RAND(n_edge_mod)
    elif attack_algorithm[0]=='stack':
        return STACK(n_edge_mod)
    elif attack_algorithm[0]=='flip':
         # degree flipping
        # attack=FLIP(n_edge_mod, flip_type="deg", mode="descend")
        # #  betweenness flipping
        # attack = FLIP(n_edge_mod, flip_type="bet", mode="ascend")
        # eigen flipping
        attack = FLIP(n_edge_mod, flip_type="eigen", mode="descend")
        return attack
    else:
        return None

def EvasionAttack(attack_algorithm,data):

    n_edge_test = data.adj[data.test_mask].getnnz()
    n_mod_ratio = 0.1
    n_edge_mod = int(n_edge_test * n_mod_ratio)

    attack = evasion_dict(attack_algorithm,n_edge_mod)
    adj_attack = attack.attack(data.adj, data.index_test)

    return adj_attack