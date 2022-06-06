'''
Author: your name
Date: 2021-01-26 22:32:57
LastEditTime: 2021-07-07 16:51:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /TrafficManipulator/extractor.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
import argparse
import sys
import csmt.feature_extractor.AfterImageExtractor.FEKitsune as Fe
from csmt.feature_extractor.AfterImageExtractor.KitsuneTools import RunFE 
import numpy as np
from scapy.all import *

if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--input_path', default='tests/example/test.pcap',type=str, required=False, help="raw traffic (.pcap) path")
    parse.add_argument('-o', '--output_path', type=str, default='tests/example/test.npy',required=False, help="feature vectors (.npy) path")

    arg = parse.parse_args()
    pcap_file = arg.input_path

    feat_file = arg.output_path

    scapyin = rdpcap(pcap_file)

    FE = Fe.Kitsune(scapyin, np.Inf)
    feature, _ = RunFE(FE)
    print(feature[100])
    print(feature[0].shape)
    print(np.asarray(feature).shape)
    np.save(feat_file,feature)

            
