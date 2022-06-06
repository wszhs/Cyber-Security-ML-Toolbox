import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.feature_extractor.cicflowmeter.flow_session import FlowSession
from scapy.all import rdpcap,wrpcap,hexdump

input_file='tests/example/test.pcap'
output_mode='flow'
output_file='tests/example/flows.csv'
url_model=None

NewFlowSession = FlowSession(output_mode, output_file, url_model)

pktList = rdpcap(input_file)
for i in range(len(pktList)):
    pkt = pktList[i]
    NewFlowSession.on_packet_received(packet=pkt)

NewFlowSession.toPacketList()