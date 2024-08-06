import time


def measure_dataloader_time(dataloader):
    start_time = time.time()
    for batch_idx, batch in enumerate(dataloader):
        pass  # Simulate processing
    end_time = time.time()
    total_time = end_time - start_time
    return total_time
