import torch
import torch.multiprocessing as mp
import os

def worker_process(rank, tensor_size):
    # Set the multiprocessing start method to 'spawn'
    # This should be done in the main section, but included here for clarity
    # mp.set_start_method('spawn', force=True)

    # Ensure each process uses the same GPU (e.g., GPU 0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create a tensor and send it to the GPU
    tensor = torch.ones(tensor_size) * rank  # Differentiate tensors by rank
    tensor = tensor.to(device)

    # Perform some computations
    result = tensor * 2

    # Print the result
    print(f"Process {rank}: Computation result on device {device}: {result}")

def main():
    # Set the multiprocessing start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    # Define the size of the tensor
    tensor_size = (1000, 1000)

    # Create two processes
    processes = []
    for rank in range(2):
        p = mp.Process(target=worker_process, args=(rank, tensor_size))
        p.start()
        processes.append(p)

    # Wait for processes to complete
    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
