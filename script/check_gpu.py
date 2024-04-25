import subprocess


def check_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    output = subprocess.check_output(command, shell=True)
    memory_free_info = output.decode('utf-8').split('\n')[1:-1]  # skip the first and last line
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return all(x > 3072 for x in memory_free_values)  # 3 GB


if __name__ == "__main__":
    if check_gpu_memory():
        exit(0)
    else:
        exit(1)
