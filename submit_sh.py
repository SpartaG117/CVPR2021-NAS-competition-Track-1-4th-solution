import sys, os
import argparse

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--output_dir', type=str, default='op', help='experiment name')
parser.add_argument('--model_name', type=str, default='weights.pth', help='path to save the model')
parser.add_argument('--start', type=int, default=0)
parser.add_argument('--end', type=int, default=50000)

parser.add_argument('--num_gpu', type=int, default=4, help='num gpu')
parser.add_argument('--proc_id', type=int, default=0)
parser.add_argument('--n_p', type=int, default=16, help='num of processes')

if __name__ == '__main__':
    args = parser.parse_args()
    num_gpu = args.num_gpu
    start = args.start
    end = args.end
    n_p = args.n_p
    output_dir = args.output_dir
    model_name = args.model_name
    sub_dir = os.path.join('./checkpoints',args.output_dir, 'submission')
    if not os.path.isdir(sub_dir):
        os.mkdir(sub_dir)
    
    sh_file_name = f'{sub_dir}/run_{start}_{end}.sh'
    with open(sh_file_name, 'w') as file_out:
        for k in range(n_p):
            one_command = f'python submit_single.py --n_p {n_p} --proc_id {k} --start {start} --end {end} --output_dir {output_dir} --model_name {model_name}'
            if k != n_p - 1:
                one_command += ' &'
            file_out.write(one_command + '\n')
        file_out.write('echo "job finished"\n')
