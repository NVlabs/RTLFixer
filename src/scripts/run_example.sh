python main.py \
  --run_name "test_oneshot_compile" \
  # Root directory for the exp run
  --root_dir "exp" \
  # Path to the dataset
  --dataset_path ./benchmarks/verilogeval-syntax-hard.jsonl \
  # Task for running [oneshot_fix_compile, react_fix_compile, react_fix_simulate]
  --task "oneshot_fix_compile" \
  # Specifying the agent to use [nefeedback, feedback, rag]
  --agent_feedback "rag" \
  --language "verilog" \
  --model "gpt-3.5-turbo-16k-0613" \
  --pass_at_k "1" \
  # Number of samples for each problem instance
  --num_samples '1' \
  # Compiler to use [iverilog, modelsim, vcs, quartus]
  --compiler 'quartus' \
  --verbose
