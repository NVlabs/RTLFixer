import os
import argparse
from task import (
    react_fix_compile,
    react_fix_simulate,
    oneshot_fix_compile,
)

from utils import read_jsonl, read_jsonl_gz


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, help="The name of the run")
    parser.add_argument("--root_dir", type=str,
                        help="The root logging directory", default="root")
    parser.add_argument("--prefix", type=str,
                        help="log file prefix", default="")
    parser.add_argument("--dataset_path", type=str,
                        help="The path to the benchmark dataset", default="root")
    parser.add_argument("--task", type=str,
                        help="Task: `react_fix_compile`, `react_fix_simulate`, `oneshot_fix_compile`")
    parser.add_argument("--agent_feedback", type=str,
                        help="Agent: `nofeedback`, `feedback`, `rag`", default="cot")
    parser.add_argument("--compiler", type=str,
                        help="Compiler: `iverilog`, `modelsim`, `vcs`, `quartus`", default="iverilog")
    parser.add_argument("--language", type=str, help="`verilog` or `py` or `rs`")
    parser.add_argument(
        "--model", type=str, help="OpenAI models only for now. For best results, use GPT-4")
    parser.add_argument("--pass_at_k", type=int,
                        help="Pass@k metric", default=1)
    parser.add_argument("--max_iters", type=int,
                        help="The maximum number of self-improvement iterations", default=10)
    parser.add_argument("--max_budgets", type=int,
                        help="The maximum number of simulation budgets", default=10)
    parser.add_argument("--num_samples", type=int,
                        help="The maximum number of sample generation", default=10)
    parser.add_argument("--verbose", action='store_true',
                        help="To print live logs")
    args = parser.parse_args()
    return args


def task_factory(task: str):
    def kwargs_wrapper_gen(func, delete_keys=[]):
        def kwargs_wrapper(**kwargs):
            for key in delete_keys:
                if key in kwargs:
                    del kwargs[key]
            return func(**kwargs)
        return kwargs_wrapper

    if task == "react_fix_compile":
        return kwargs_wrapper_gen(react_fix_compile, delete_keys=[])
    elif task == "react_fix_simulate":
        return kwargs_wrapper_gen(react_fix_simulate, delete_keys=["compiler"])
    elif task == "oneshot_fix_compile":
        return kwargs_wrapper_gen(oneshot_fix_compile, delete_keys=["max_budgets", 'max_iters'])
    else:
        raise ValueError(f"Task `{task}` is not supported")


def main(args):
    # check if the root dir exists and create it if not
    if not os.path.exists(args.root_dir):
        os.makedirs(args.root_dir)

    # get the dataset name
    dataset_name = os.path.basename(args.dataset_path).replace("jsonl", "")

    # check if log path already exists
    log_dir = os.path.join(args.root_dir, args.run_name)
    log_path = os.path.join(
        log_dir, f"{args.prefix}_{dataset_name}_{args.task}_iter_{args.max_iters}_num_sample_{args.num_samples}_{args.agent_feedback}_{args.model}_pass_at_k_{args.pass_at_k}_{args.language}_{args.compiler}.jsonl")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # check if the task is valid
    run_task = task_factory(args.task)

    # print starting message
    if args.verbose:
        print(f"""
Starting run with the following parameters:
task: {args.task}
agent_feedback: {args.agent_feedback}
num_samples: {args.num_samples}
pass@k: {args.pass_at_k}
""")
    else:
        print(f"Logs will be saved in `{log_dir}`")

    # load the dataset
    print(f'Loading the dataset...')
    if args.dataset_path.endswith(".jsonl"):
        dataset = read_jsonl(args.dataset_path)
    elif args.dataset_path.endswith(".jsonl.gz"):
        dataset = read_jsonl_gz(args.dataset_path)
    else:
        raise ValueError(
            f"Dataset path `{args.dataset_path}` is not supported")

    print(f"Loaded {len(dataset)} examples")
    # start the run
    # evaluate with pass@k
    run_task(
        dataset=dataset,
        model_name=args.model,
        agent_feedback=args.agent_feedback,
        language=args.language,
        max_iters=args.max_iters,
        max_budgets=args.max_budgets,
        pass_at_k=args.pass_at_k,
        log_path=log_path,
        verbose=args.verbose,
        num_samples=args.num_samples,
        compiler=args.compiler,
    )

    print(f"Done! Check out the logs in `{log_path}`")
    
    if args.language == "verilog":
        from verilog_eval.execution import clean_up_simulation
        clean_up_simulation()


if __name__ == "__main__":
    args = get_args()
    main(args)
