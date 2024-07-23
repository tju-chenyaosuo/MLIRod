import re
import os
import argparse
import random


def get_file_lines(file_path):
    f = open(file_path)
    lines = f.readlines()
    return lines


def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()


def execmd(cmd):
    import os
    print('[execmd] ' + cmd)
    try:
        pipe = os.popen(cmd)
        reval = pipe.read()
        pipe.close()
        return reval
    except BlockingIOError:
        print("[execmd] trigger BlockingIOError")
        return "None"

def execmd_limit_time(cmd, time_limit):
    import time
    start = time.time()
    execmd("timeout " + str(time_limit) + " " + cmd)
    end = time.time()
    return (end - start) >= time_limit


def not_mlir(file):
    cmd = f"file -r {file}"
    ret = execmd(cmd)
    print(ret)
    return ": ASCII text" not in ret


def get_line_number(file):
    return len(get_file_lines(file))
    # res = execmd("wc -l " + file)
    # res = res[:-1] if res.endswith("\n") else res
    # res = res.split(' ')[0]
    # return int(res)


def random_file_prefix():
    # pattern: "/tmp/tmp.WeWmILaF2A"
    cmd = "mktemp -p " + args.seed_dir
    random_name = execmd(cmd)
    random_name = random_name[:-1] if random_name.endswith('\n') else random_name
    res_name = random_name.split('/')[-1]
    cmd = "rm -f " + random_name
    os.system(cmd)
    return res_name


def diff(file1, file2):
    return execmd(f"diff {file1} {file2}")


class Seed:
    def __init__(self, path):
        self.path = path
        self.cnt = 0
        self.tested = False
        self.deprecated = False

    def can_discard(self):
        return self.cnt >= args.max_selected or self.deprecated

    def not_mlir(self):
        return not_mlir(self.path)

    def __str__(self):
        return f'seed: [path={self.path}, mutation_cnt={self.cnt}, is_tested={self.tested}]'


class ExistingSeedFactory:
    def __init__(self):
        self.existing_mlirs = os.listdir(args.existing_seed_dir)
        self.existing_mlirs = [args.existing_seed_dir + os.sep + _ for _ in self.existing_mlirs]
        self.unselected_idx = [_ for _ in range(len(self.existing_mlirs))]
        pass

    def gen_seed(self):
        tmp = random.randint(0, len(self.unselected_idx) - 1)
        idx = self.unselected_idx[tmp]
        del self.unselected_idx[tmp]

        rname = random_file_prefix() + '.mlir'
        seed_path = args.seed_dir + os.sep + rname
        cmd = f'cp {self.existing_mlirs[idx]} {seed_path}'
        os.system(cmd)
        s = Seed(seed_path)

        return s


class MLIRSmithSeedFactory:
    def __init__(self):
        self.conf_success = "conf_success"
        self.conf_timeout = "conf_timeout"
        self.conf_crash = "conf_crash"
        self.mlir_success = "mlir_success"
        self.mlir_timeout = "mlir_timeout"
        self.mlir_crash = "mlir_crash"

    def gen_conf(self):
        # generate new configuration
        cmd = args.mlirsmith + " -emit=config >" + args.conf
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.conf_timeout
        conf_lines_len = get_line_number(args.conf)
        if conf_lines_len < 100:
            return self.conf_crash
        return self.conf_success

    def gen_mlir(self, mlir_path):
        cmd = args.mlir_smith + " -emit=mlir-affine " + " " + args.mlir_template + " 2>" + mlir_path
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mlir_timeout
        mlir_file_len = get_line_number(mlir_path)
        if mlir_file_len < 100:
            return self.mlir_crash
        return self.mlir_success

    def gen_seed(self):
        # generate a new conf
        res = self.gen_conf()
        while res != self.conf_success:
            res = self.gen_conf()
        # gen random mlir file
        mlir_name = random_file_prefix() + '.mlir'
        mlir_path = args.seed_dir + os.sep + mlir_name
        # gen mlir file
        res = self.gen_mlir(mlir_path)
        while res != self.mlir_success:
            res = self.gen_mlir(mlir_path)
        s = Seed(mlir_path)
        return s


class SeedPool:
    def __init__(self):
        self.pool = []

    def init(self):
        print(f'[SeedPool.init] the pool initializer is: {args.seed}')
        factory = None
        if args.seed == "mlirsmith":
            factory = MLIRSmithSeedFactory()
        if args.seed == "existing":
            factory = ExistingSeedFactory()

        assert factory is not None
        while len(self.pool) < args.init_size:
            self.add(factory.gen_seed())
        print('[SeedPool.init] Seed pool initialization is over!')
        print('[SeedPool.init] The seed information is:')
        for s in self.pool:
            print(s)

    def select(self):
        idx = random.randint(0, len(self.pool) - 1)
        selected = self.pool[idx]
        while selected.not_mlir():
            del selected[idx]
            idx = random.randint(0, len(self.pool) - 1)
            selected = self.pool[idx]
        return selected

    def add(self, s : Seed):
        if not s.not_mlir():
            print('[SeedPool.add] new seed: ' + str(s))
            self.pool.append(s)

    def empty(self):
        return len(self.pool) == 0

    def clean(self):
        idx = 0
        while idx < len(self.pool):
            if self.pool[idx].can_discard():
                del self.pool[idx]
            else:
                idx += 1

    def __str__(self):
        res = f"pool size: {len(self.pool)}\n"
        for l in self.pool:
            res += f'{l}\n'
        return res


class Mutator:
    def __init__(self, in_seed: Seed):
        self.in_seed = in_seed
        self.out = None
        self.mutation_timeout = "mutation_timeout"
        self.mutation_crash = "mutation_crash"
        self.mutation_success = "mutation_success"
        self.mutation_failed = "mutation_failed"
        self.mutators = [self.replace_data_edge, self.replace_control_edge,
                         self.delete_node, self.create_node]
        # self.mutators = [self.replace_data_edge, self.replace_control_edge,
        #                  self.delete_node]

    def replace_data_edge(self):
        random_name = random_file_prefix() + '.mlir'
        new_mlir = args.seed_dir + os.sep + random_name
        self.out = new_mlir
        cmd = ' '.join([args.replace_data_edge, "--operand-prob", str(args.operand_prob), self.in_seed.path, new_mlir])
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mutation_timeout
        if not os.path.exists(new_mlir) or get_line_number(new_mlir) < 100:
            return self.mutation_crash
        return self.mutation_success

    def replace_control_edge(self):
        random_name = random_file_prefix() + '.mlir'
        new_mlir = args.seed_dir + os.sep + random_name
        self.out = new_mlir
        cmd = ' '.join([args.replace_control_edge, self.in_seed.path, new_mlir])
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mutation_timeout
        if not os.path.exists(new_mlir) or not_mlir(new_mlir):
            return self.mutation_crash
        return self.mutation_success

    def delete_node(self):
        random_name = random_file_prefix() + '.mlir'
        new_mlir = args.seed_dir + os.sep + random_name
        self.out = new_mlir
        cmd = ' '.join([args.delete_node, self.in_seed.path, new_mlir])
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mutation_timeout
        if not os.path.exists(new_mlir) or not_mlir(new_mlir):
            return self.mutation_crash
        return self.mutation_success

    def create_node(self):
        random_name = random_file_prefix() + '.mlir'
        new_mlir = args.seed_dir + os.sep + random_name
        self.out = new_mlir
        cmd = ' '.join([args.create_node, self.in_seed.path, new_mlir])
        timeout = execmd_limit_time(cmd, args.timeout)
        if timeout:
            return self.mutation_timeout
        if not os.path.exists(new_mlir) or not_mlir(new_mlir):
            return self.mutation_crash
        return self.mutation_success

    def mutate(self):
        mutation = self.mutators[random.randint(0, len(self.mutators) - 1)]
        print(f'[Mutator.mutate] Selected mutator: {mutation}')
        res = mutation()
        print(f'[Mutator.mutate] Mutation result: {res}')
        self.in_seed.cnt += 1
        return res


class OptimizationSelector:
    def __init__(self):
        self.opts = get_file_lines(args.optfile)
        self.opts = [_[:-1] if _.endswith('\n') else _ for _ in self.opts]
        self.opt_effectiveness = {_:{} for _ in self.opts}  # effectiveness[opt][operation] = [work_time, accur_time]
        pass

    def select_idx(self):
        return random.randint(0, len(self.opts) - 1)

    def get_option(self, idx):
        return self.opts[idx]

    def collect_operation(self, mlir_file):
        # match xxx.xxx
        operations = {}
        print(mlir_file)
        file_lines = get_file_lines(mlir_file)
        for line in file_lines:
            pattern = re.compile(r'[a-zA-Z][a-zA-Z0-9]+\.[a-zA-Z0-9]+')
            m = pattern.findall(line)
            if m is None or len(m) == 0:
                continue
            operations[m[0]] = operations.setdefault(m[0], 1) + 1
        return operations

    def update_effectiveness(self, opt, mlir_before, mlir_after):
        # collect operation count before transformation
        operations_before = self.collect_operation(mlir_before)
        # collect operation count after transformation
        operations_after = self.collect_operation(mlir_after)
        # calculate effective operation-opt pair, and update effectiveness information
        for _ in operations_before:
            # update occur time.
            self.opt_effectiveness[opt].setdefault(_, [1, 1])[1] += 1
            if operations_after.setdefault(_, 0) - operations_before[_] < 0:
                # update work time
                self.opt_effectiveness[opt][_][0] += 1

    def select(self, mlir_file):
        # selected_opts = []
        # for _ in range(10):
        #     rand_len = random.randint(1, 10)
        #     opt = " ".join([self.opts[random.randint(0, len(self.opts)-1)] for _ in range(rand_len)])
        #     selected_opts.append(opt)
        # return selected_opts

        selected_opts = [self.opts[random.randint(0, len(self.opts)-1)] for _ in range(10)]
        return selected_opts
        # operations = self.collect_operation(mlir_file)
        # selected_opts = []
        # for opt in self.opts:
        #     occur_time = 0
        #     work_time = 0
        #     for operand in operations:
        #         if operand not in self.opt_effectiveness[opt]:
        #             occur_time += 1
        #             work_time += 1
        #         else:
        #             work_time += self.opt_effectiveness[opt][operand][0]
        #             occur_time += self.opt_effectiveness[opt][operand][1]
        #     if float(random.randint(0, 100)) / 100 < float(work_time) / occur_time:
        #         selected_opts.append(opt)
        # return selected_opts


class Tester:
    def __init__(self, seed: Seed, selector: OptimizationSelector):
        self.seed = seed
        self.optimization_selector = selector
        self.report_dir = args.report_dir + os.sep + os.path.basename(seed.path)
        os.system('mkdir -p ' + self.report_dir)
        pass

    def execute_and_retention(self, seed_pool: SeedPool):
        work = False
        selected_options = self.optimization_selector.select(self.seed.path)
        for option_idx in range(len(selected_options)):
            report = self.report_dir + os.sep + str(option_idx) + '.crash.txt'
            option = selected_options[option_idx]
            rname = random_file_prefix() + '.mlir'
            out_mlir = args.seed_dir + os.sep + rname

            # /data1/src/AFLplusplus-4.09c/afl-showmap -o test.map -t 3000 -- {your commands}

            cmd = ' '.join([args.mlir_opt, option, self.seed.path, '-o', out_mlir, '2>', report])
            cmd = f'{args.collector} -o {os.getpid()}.map -t {args.timeout*1000} -- {cmd}'
            execmd_limit_time(cmd, args.timeout)
            put_file_content(report, cmd)

            cond1 = os.path.exists(out_mlir)
            cond2 = False
            if cond1:
                cond2 = not not_mlir(out_mlir)
            cond3 = False
            if cond1 and cond2:
                cond3 = get_file_lines(self.seed.path) != get_file_lines(out_mlir)

            if cond1 and cond2 and cond3:
                # update effective optimizations
                # self.optimization_selector.update_effectiveness(option, self.seed.path, out_mlir)
                # update the seed pool.
                c = Coverage(out_mlir)
                if c.collect_and_compare():
                    work = True
                    s = Seed(out_mlir)
                    seed_pool.add(s)
                else:
                    os.system(f'rm -f {out_mlir}')
        self.seed.tested = True
        if not work:
            self.seed.deprecated = True


class Coverage:
    COVERAGE = set()

    def __init__(self, mlir_path):
        self.mlir_path = mlir_path
        pass

    def collect_cov(self):
        f_lines = get_file_lines(f'{os.getpid()}.map')
        f_cov = set([_.split(':')[0] for _ in f_lines if len(_) > 1])
        return f_cov
        # cmd = ' '.join([args.collector, self.mlir_path, "--depth", str(args.cov_depth), args.cov_file])
        # print(f'[Coverage.collect_cov] {cmd}')
        # os.system(cmd)
        # cov = set()
        # if os.path.exists(args.cov_file):
        #     cov = set(get_file_lines(args.cov_file))
        # os.system(f"rm -f {args.cov_file}")
        # return cov

    def collect_and_compare(self):
        lines_before = len(Coverage.COVERAGE)
        cov = self.collect_cov()
        Coverage.COVERAGE |= cov
        lines_after = len(Coverage.COVERAGE)
        assert lines_after >= lines_before
        update = lines_before < lines_after
        print("[Coverage.collect_and_compare] coverage length=" + str(len(Coverage.COVERAGE)))
        return update


def main():
    # seed pool initialization
    print('[main] Start fuzzing process')
    seed_pool = SeedPool()
    print('[main] Start pool initialization')
    seed_pool.init()
    print('[main] End pool initialization')
    print('[main] *************************************')

    optimizer_selector = OptimizationSelector()

    while not seed_pool.empty():
        print(f'[main] seed pool size: {len(seed_pool.pool)}')
        # print(f'[main] seed information: ')
        # print(seed_pool)

        mutate_success = False
        new_mlir = None
        new_seed = None
        old_seed = None
        while not mutate_success:
            # seed selection
            s = seed_pool.select()
            old_seed = s
            print(f'[main] Seed selection: {s}')
            if not s.tested:
                t = Tester(s, optimizer_selector)
                t.execute_and_retention(seed_pool)
            print('[main] *************************************')

            # seed mutation
            print(f'[main] Seed mutation')
            m = Mutator(s)
            res = m.mutate()
            if res != m.mutation_success:
                continue
            else:
                mutate_success = True
            new_seed = Seed(m.out)
            new_mlir = m.out
            print('[main] *************************************')

        assert new_mlir is not None
        assert new_seed is not None

        # seed retention
        print(f'[main] Seed retention')
        if len(diff(new_seed.path, old_seed.path)) > 10:
            c1 = Coverage(new_mlir)
            if c1.collect_and_compare():
                seed_pool.add(new_seed)
            else:
                os.system(f'rm -f {new_mlir}')
                print(f'[main] new seed does not have new coverage')
                continue
            print('[main] *************************************')

            # seed execution
            print(f'[main] Seed execution')
            if not new_seed.tested:
                t = Tester(new_seed, optimizer_selector)
                t.execute_and_retention(seed_pool)
            print('[main] *************************************')

        # clean the seed pool
        # seed_pool.clean()


"""
Usage:
跑实验用
timeout 86400 python mlirfuzz.py \
--seed existing \
--existing_seed_dir existing_seeds/ \
--replace_data_edge /data1/src/mlirfuzz-edge-1/build/bin/ReplaceDataEdge \
--replace_control_edge /data1/src/mlirfuzz-edge-1/build/bin/ReplaceControlEdge \
--delete_node /data1/src/mlirfuzz-edge-1/build/bin/DeleteNode \
--create_node /data1/src/mlirfuzz-edge-1/build/bin/CreateNode \
--mlir_opt /data1/src/mlirfuzz-edge-1/build/bin/mlir-opt \
--optfile opt.txt \
--collector /data1/src/AFLplusplus-4.09c/afl-showmap \
--clean True > log.txt
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # OOA arguments
    parser.add_argument("--timeout", type=int, help="Size of seed pool.", default=10)
    parser.add_argument("--clean", type=bool, help="Clean", default=False)
    # arguments in seed pool initialization
    parser.add_argument("--init_size", type=int, help="Size of seed pool.", default=50)
    parser.add_argument("--seed_dir", type=str, help="Seed directory.", default="seeds")
    parser.add_argument("--seed", type=str, help="Method to initialize the seed pool", default="mlirsmith")
    # Seed pool initialization from existing seeds.
    parser.add_argument("--existing_seed_dir", type=str, help="The path of existing seed directory", default="ori_seeds")
    # Seed pool initialization from mlirsmith.
    parser.add_argument("--mlirsmith", type=str, help="Path of mlirsmith", default="")
    parser.add_argument("--conf", type=str, help="Path of configuration file", default="")
    parser.add_argument("--mlir_template", type=str, help="Path of mlir template", default="")
    # arguments in seed selection
    None
    # arguments in seed mutation
    parser.add_argument("--replace_data_edge", type=str, help="Path of mutator OperandsReplacement", default="")
    parser.add_argument("--replace_control_edge", type=str, help="Path of mutator MoveToOutter", default="")
    parser.add_argument("--delete_node", type=str, help="Path of mutator DeleteOperation", default="")
    parser.add_argument("--create_node", type=str, help="Path of mutator DeleteOperation", default="")
    parser.add_argument("--operand_prob", type=float, help="Probability for operands to be selected to mutate", default=0.1)
    # arguments in seed execution
    parser.add_argument("--mlir_opt", type=str, help="Path of mlir_opt", default="")
    parser.add_argument("--optfile", type=str, help="Path of optimization file", default="")
    parser.add_argument("--report_dir", type=str, help="Path of report", default="report")
    # arguments in seed retention
    parser.add_argument("--collector", type=str, help="Coverage collector", default="")
    parser.add_argument("--cov_file", type=str, help="Coverage file.", default="coverage.txt")
    parser.add_argument("--cov_depth", type=int, help="Coverage depth", default=2)
    parser.add_argument("--max_selected", type=int, help="Max selected count for each seed.", default=10)
    args = parser.parse_args()

    if args.clean:
        # os.system(f'rm -rf {args.seed_dir} {args.report_dir} {args.cov_file}')
        os.system(f'rm -rf {args.seed_dir} {args.report_dir}')

    assert args.seed in ['existing', 'mlirsmith']
    assert os.path.exists(args.existing_seed_dir)
    os.system(f'mkdir -p {args.seed_dir}')
    os.system(f'mkdir -p {args.report_dir}')
    os.system(f'touch {args.cov_file}')

    main()


