'''
Author: your name
Date: 2021-06-24 12:57:50
LastEditTime: 2021-07-07 20:01:17
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /Cyber-Security-ML-Toolbox/tests/feature_extractor/test_flowmeter.py
'''
import sys
sys.path.append("/Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox")
from csmt.feature_extractor.cicflowmeter.sniffer import create_sniffer
import argparse

def main_old():
    parser = argparse.ArgumentParser()

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--interface",
        action="store",
        dest="input_interface",
        help="capture online data from INPUT_INTERFACE",
    )

    input_group.add_argument(
        "-f",
        "--file",
        action="store",
        dest="input_file",
        help="capture offline data from INPUT_FILE",
        default='tests/example/test.pcap'
    )

    output_group = parser.add_mutually_exclusive_group(required=False)
    output_group.add_argument(
        "-c",
        "--csv",
        "--flow",
        action="store_const",
        const="flow",
        dest="output_mode",
        help="output flows as csv",
        default='tests/example/flows.csv'
    )

    url_model = parser.add_mutually_exclusive_group(required=False)
    url_model.add_argument(
        "-u",
        "--url",
        action="store",
        dest="url_model",
        help="URL endpoint for send to Machine Learning Model. e.g http://0.0.0.0:80/prediction",
    )

    parser.add_argument(
        "output",
        help="output file name (in flow mode) or directory (in sequence mode)"
    )

    args = parser.parse_args()

    # print(args.input_file,
    #     args.input_interface,
    #     args.output_mode,
    #     args.output,
    #     args.url_model)
    sniffer = create_sniffer(
        args.input_file,
        args.input_interface,
        args.output_mode,
        args.output,
        args.url_model,
    )
    sniffer.start()

    try:
        sniffer.join()
    except KeyboardInterrupt:
        sniffer.stop()
    finally:
        sniffer.join()


def main():
    input_file='tests/example/test2.pcap'
    input_interface=None
    output_mode='flow'
    output='tests/example/flows.csv'
    url_model=None
    
    sniffer = create_sniffer(
        input_file,
        input_interface,
        output_mode,
        output,
        url_model
    )
    sniffer.start()

    try:
        sniffer.join()
    except KeyboardInterrupt:
        sniffer.stop()
    finally:
        sniffer.join()


if __name__ == "__main__":
    main()
# python test_flowmeter.py -f /Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/tests/example/test.pcap  -c /Users/zhanghangsheng/Documents/my_code/Cyber-Security-ML-Toolbox/tests/example/flows.csv