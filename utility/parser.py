import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="NIE_GCN")

    parser.add_argument("--seed", type=int, default=2020, help="random seed for init")

    parser.add_argument("--dataset", nargs="?", default="yelp2018", help="dataset")

    parser.add_argument("--data_path", nargs="?", default="./Data/", help="Input data path.")

    parser.add_argument("--model", default="NIE_GCN", help="Model to train")

    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=1024, help='train batch size')

    parser.add_argument("--layer_att", type=int, default=1, help='use layer_attention or not')

    parser.add_argument('--GCNLayer', type=int, default=3, help="the layer number of GCN")

    parser.add_argument("--agg", default="sum", help="[cat, sum]")

    parser.add_argument('--beta', type=float, default=1.0, help='beta for softmax')
                        
    parser.add_argument('--test_batch_size', type=int, default=200, help='test batch size')

    parser.add_argument('--dim', type=int, default=64, help='embedding size')

    parser.add_argument('--l2', type=float, default=1e-4, help='l2 regularization weight')

    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")

    parser.add_argument('--topK', nargs='?', default='[20, 40, 60]', help='size of Top-K')

    parser.add_argument('--test_flag', nargs='?', default='part', help='Specify the test type from {part, full}')

    parser.add_argument("--verbose", type=int, default=10, help="Test interval")

    parser.add_argument("--multicore", type=int, default=0, help="use multiprocessing or not in test")

    return parser.parse_args()
