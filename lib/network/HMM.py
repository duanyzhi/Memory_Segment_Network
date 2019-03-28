from lib.config.config import FLAGS as cfg
import math


class HMM(object):
    def __init__(self):
        pass

    def __call__(self, baysian_information):
        pre_bays = baysian_information[0][:cfg.N]
        acc = 0
        if pre_bays[0][0] <= cfg.loc_threshold:
            acc += 1
        for loc, bays in enumerate(baysian_information[1:]):
            # pre_bays = baysian_information[loc][:cfg.N]
            # print("pre_bays", pre_bays)
            pre_bays = self.norm(pre_bays)
            # print("bays", bays)
            one_location_new_pi = []
            for bay in bays[:cfg.N]:
                new_pro = 0
                for pre_bay in pre_bays:
                    # print("pre_bay", pre_bay)
                    G = self.Gaussian(bay[0], pre_bay[0])
                    # print("G", G)
                    new_pro += pre_bay[1]*G*bay[1]
                one_location_new_pi.append([bay[0], new_pro])
            # print("one_location_new_pi", one_location_new_pi)
            pre_bays = one_location_new_pi[::]
            one_location_new_pi.sort(key=lambda f:f[1], reverse=True)
            # print("sort one loc", one_location_new_pi)
            # print("pre_bays", pre_bays)
            if abs(one_location_new_pi[0][0] - loc -1) <= cfg.loc_threshold:
                acc += 1
            print("real location:", loc + 1, "pre location:", one_location_new_pi[0][0])
        print("Memory Slice Network with HMM Accaury:", acc / (loc + 1))
        return acc / (loc + 1)

    def norm(self, pro_list):
        # print(pro_list)
        max_value = max(pro_list, key=lambda f:f[1])[1] + 0.000001
        return [[p[0], p[1]/max_value] for p in pro_list]

    def Gaussian(self, l1, l2):
        return (1 / (math.sqrt(2 * math.pi) * cfg.sigma)) * math.exp(
            -1 * ((l1 - l2) ** 2) / (2 * math.pi * (cfg.sigma ** 2)))
