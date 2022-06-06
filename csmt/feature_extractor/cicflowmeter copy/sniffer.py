'''
Author: your name
Date: 2021-06-24 11:35:59
LastEditTime: 2021-07-07 20:00:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/csmt/feature_extractor/cicflowmeter/sniffer.py
'''

from scapy.sendrecv import AsyncSniffer

from csmt.feature_extractor.cicflowmeter.flow_session import generate_session_class


def create_sniffer(
    input_file, input_interface, output_mode, output_file, url_model=None
):
    assert (input_file is None) ^ (input_interface is None)

    NewFlowSession = generate_session_class(output_mode, output_file, url_model)

    if input_file is not None:
        return AsyncSniffer(
            offline=input_file,
            filter="ip and (tcp or udp)",
            prn=None,
            session=NewFlowSession,
            store=False,
        )
    else:
        return AsyncSniffer(
            iface=input_interface,
            filter="ip and (tcp or udp)",
            prn=None,
            session=NewFlowSession,
            store=False,
        )

