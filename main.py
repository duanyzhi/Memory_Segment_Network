from lib.network.net import msn
import argparse

def run(pattern, data):
    net = msn(pattern, data)
    net.build_net()
    if pattern == "train":
        net.BP()
    net.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pattern', type=str, default='test',  # pattern前面加两个--表示pattern是可选参数
                        required=True, help='Choice train or test model')
    parser.add_argument('--data', type=str, default='google',
                        required=True, help='Choice which dataset')

    args = parser.parse_args()

    assert args.data == 'google' or args.data == 'oxford' or args.data == 'uestc'

    run(args.pattern, args.data) # train or test

# python main.py --pattern train --data google
# python main.py --pattern test --data google
