import os
import re
import random
import argparse
from enum import Enum


def execmd(cmd):
    import os
    print('[execmd] ' + cmd)
    pipe = os.popen(cmd)
    reval = pipe.read()
    pipe.close()
    return reval


def random_file_prefix():
    # pattern: "/tmp/tmp.WeWmILaF2A"
    cmd = "mktemp -p " + args.testcase_base
    random_name = execmd(cmd)
    random_name = random_name[:-1] if random_name.endswith('\n') else random_name
    res_name = random_name.split('/')[-1]
    cmd = "rm -f " + random_name
    os.system(cmd)
    return res_name

def execmd_limit_time(cmd, time_limit):
    import time
    start = time.time()
    execmd("timeout " + str(time_limit) + " " + cmd)
    end = time.time()
    return (end - start) >= time_limit


def get_file_content(file_path):
    f = open(file_path, 'r')
    content = f.read()
    f.close()
    return content


def get_file_lines(path):
    c = get_file_content(path)
    if c == '':
        return ''
    if c[-1] == '\n':
        return c[:-1].split('\n')
    else:
        return c.split('\n')


def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()


class TestState(Enum):
    # config_timeout = 0
    # config_crash = 1
    # config_success = 2
    config_success = 0
    config_failed = 1
    gen_success = 3
    gen_timeout = 4
    gen_crash = 5
    test_success = 6
    test_timeout = 7
    test_crash = 8


def gen_one_mlir(mlir_file):
    gen_cmd = f"{args.mlirsmith} 2>{mlir_file}"
    timeout = execmd_limit_time(gen_cmd, args.timeout)
    if timeout:
        return TestState.gen_timeout
    f = open(mlir_file)
    f_lines = f.readlines()
    f.close()
    if len(f_lines) < 100:
        return TestState.gen_crash
    return TestState.gen_success


def test_each_mlir(mlir_file, opt, out_mlir, report_dir):
    crash_log = report_dir + os.sep + os.path.basename(out_mlir) + '.crash.txt'
    cmd = ' '.join([args.mliropt, opt, mlir_file, '1>/dev/null', '2>' + crash_log])
    timeout = execmd_limit_time(cmd, 10)
    if timeout:
        timeout_log = report_dir + os.sep + os.path.basename(out_mlir) + '.timeout.txt'
        put_file_content(timeout_log, cmd)
        os.system('rm -f ' + crash_log)
        return TestState.test_timeout
    log_content = get_file_content(crash_log)
    crash = len(log_content) > 10
    if crash:
        put_file_content(crash_log, '\n' + cmd + '\n')
        return TestState.test_crash
    else:
        os.system('rm -f ' + crash_log)
    return TestState.test_success

def test_each_option(mlir_file, options):
    if args.optnum < 100:
        return options
    else:
        # create temp directory
        temp_dir = execmd('mktemp -d')
        temp_dir = temp_dir[:-1] if temp_dir.endswith('\n') else temp_dir

        # for each option, we do test
        valid_options = []
        for idx in range(len(options)):
            out_mlir = temp_dir + os.sep + str(idx) + '.mlir'
            report_dir = temp_dir + os.sep + 'report'
            os.system('mkdir -p ' + report_dir)
            test_state = test_each_mlir(mlir_file, options[idx], out_mlir, report_dir)
            if test_state == TestState.test_success:
                valid_options.append(options[idx])

        os.system('rm -rf ' + temp_dir)
        return valid_options


def get_opts():
    assert os.path.exists(args.optfile)
    f = open(args.optfile)
    f_lines = f.readlines()
    f.close()
    options = [_[:-1] if _.endswith('\n') else _ for _ in f_lines]
    return options


def get_optimization_sequences(options):
    option_number = random.randint(5, 10)
    return [' '.join([options[random.randint(0, len(options) - 1)]
                      for _ in range(option_number)])
            for __ in range(args.optnum)]


def main():
    options = get_opts()
    while True:
        mf = args.testcase_base + os.sep + random_file_prefix() + '.mlir'
        res = gen_one_mlir(mf)
        while res != TestState.gen_success:
            res = gen_one_mlir(mf)
        valid_options = test_each_option(mf, options)
        if len(valid_options) == 0:
            continue
        optimization_sequences = get_optimization_sequences(valid_options)
        print('[test_mlir] valid option length: ' + str(len(valid_options)))
        print('[test_mlir] valid option ' + ', '.join(valid_options))
        for idx in range(len(optimization_sequences)):
            out_dir = args.out_base + os.sep + os.path.basename(mf) + '.output'
            report_dir = out_dir + os.sep + 'report'
            os.system('mkdir -p ' + out_dir)
            os.system('mkdir -p ' + report_dir)
            out_mlir = out_dir + os.sep + str(idx) + '.mlir'
            state = test_each_mlir(mf, optimization_sequences[idx], out_mlir, report_dir)
            # if state != TestState.test_success:
            os.system('rm -f ' + out_mlir)


"""
测试开发版本用的
timeout 86400 python mlirsmith.py \
  --mlirsmith /data/exp/2023.11.19.final.exp/src/mlirsmith/build/bin/mlirsmith \
  --testcase_base testcase \
  --mliropt /data/exp/2023.11.19.final.exp/src/mlirsmith/build/bin/mlir-opt \
  --optfile opt.txt \
  --optnum 1000 \
  --out_base report > log

跑实验用的
timeout 86400 python mlirsmith.py \
  --mlirsmith /data/exp/2023.11.19.final.exp/src/mlirsmith/build/bin/mlirsmith \
  --testcase_base testcase \
  --mliropt /data/exp/2023.11.19.final.exp/src/mlirsmith/build/bin/mlir-opt \
  --optfile opt.txt \
  --optnum 1000 \
  --out_base report > log
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlirsmith", type=str, help="path of mlirsmith.")
    parser.add_argument("--testcase_base", type=str, help="path of generated testcase directory.")
    parser.add_argument("--timeout", type=int, default=10, help="path of generated testcase directory.")
    parser.add_argument("--mliropt", type=str, help="path of mlir-opt.")
    parser.add_argument("--optfile", type=str, help="path of option file.")
    parser.add_argument("--optnum", type=int, help="number of options.")
    parser.add_argument("--out_base", type=str, help="path of output.")
    args = parser.parse_args()

    main()
