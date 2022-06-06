import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import random
import string
import numpy as np
import pandas
from scapy.all import rdpcap,wrpcap,hexdump
from sympy import print_ccode
import csmt.feature_extractor.AfterImageExtractor.FEKitsune as Fe
from csmt.feature_extractor.AfterImageExtractor.KitsuneTools import RunFE 
import numpy as np
from scapy.all import *

def random_bytes(length):
    tmp_str = ''.join(random.choice(string.printable) for _ in range(length))
    return bytes(tmp_str, encoding='utf-8')

def mal_fea(new_pcap):
    new_fe=Fe.Kitsune(new_pcap, np.Inf)
    new_feature, _ = RunFE(new_fe)
    new_feature=np.array(new_feature)
    return new_feature

def mal_pcap_evasion(models):
    mal_pcap_file='tests/example/malware.pcap'
    mal_scapy = rdpcap(mal_pcap_file)

    new_scapy=copy.deepcopy(mal_scapy)

    # new_scapy[3]=new_scapy[1]
    new_scapy[0].time=1

    # begin_time = float(mal_scapy[0].time)
    # end_time = float(mal_scapy[19].time)
    # print(mal_scapy[0].payload)
    # print(mal_scapy[0].dst)

    # new_scapy[0].add_payload(random_bytes(round(20)))
    # new_scapy[1].add_payload(random_bytes(round(20)))
    # new_scapy[2].add_payload(random_bytes(round(20)))

    # wrpcap("tests/example/new.pcap",new_scapy)

    # mal_fe=Fe.Kitsune(mal_scapy, np.Inf)
    # mal_feature, _ = RunFE(mal_fe)
    # mal_feature=np.array(mal_feature)


mal_file='csmt/datasets/data/Kitsune/mirai/Mirai_pcap.pcap'
pktList = rdpcap(mal_file)

# print(pktList.summary())
# print(pktList.nsummary())
# print(pktList.show())
# for i in range(len(pktList)):
#     print(hexdump(pktList[i]))

print("read %d packets in pcap" % (len(pktList)))
begin_timestamp = float(pktList[0].time)
last_end_time = float(pktList[9].time)
# print(pktList[0].payload)
# print(pktList[0].dst)

# print(last_end_time-begin_timestamp)
# print(last_end_time)
# print(begin_timestamp)

wrpcap("tests/example/test2.pcap",pktList[0:10])
# wrpcap("tests/example/malware2.pcap",pktList[130000:230000])

# pktList[0].add_payload(random_bytes(round(20)))


