from ultralytics import YOLO
import multiprocessing
import time
import os
import random
import json
from tqdm import tqdm

def process_image(imageName, model, i, confidence=0.4):
    latency = []
    # Load the image
    for image in imageName:
        imagePath = os.path.join(r"D:\imagenet", image)
        start_time = time.time()
        results = model(imagePath, conf=confidence, verbose=False)
        end_time = time.time() - start_time
        latency.append(end_time)
        # Print latency for each image
        # print(f"Latency of process {i} in image {imagePath} is {end_time}")
    return sum(latency)

def main(images, num_processes, imagCollection):
    # Initialize YOLO model
    qnLinemodel = YOLO("yolov8n.pt")
    # List to store latencies of each process
    latencies = []
    
    # Create a pool of processes
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Process each image using multiprocessing
    for i in range(num_processes):
        imageToPass = images[imagCollection * i:(imagCollection) * (i + 1)]
        latency = pool.apply_async(process_image, args=(imageToPass, qnLinemodel, i))
        latencies.append(latency)
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Get maximum latency for each process
    max_latencies = [latency.get() for latency in latencies]
    max_latency = max(max_latencies)
    d = {
        "num_processes": num_processes,
        "imagCollection": imagCollection,
        "max_latency": max_latency
    }
    # l = []
    # l.append(d)
    return d

if __name__ == "__main__":
    origImages = os.listdir(r"D:\imagenet")  # List of image paths
    runs = 1000
    results = []

    # Run the experiment for 1000 times
    for _ in tqdm(range(runs), desc="Experiment Progress"):
        # Randomly select number of processes and imagCollection
        num_processes = random.randint(1, 30)
        imagCollection = random.randint(10, 100)
        if _ == 0:
            print(f"Current number of processes={num_processes} and number of images in each process = {imagCollection}")
        else:
            print(f"\rCurrent number of processes={num_processes} and number of images in each process = {imagCollection}")
        result = main(origImages, num_processes, imagCollection)
        # results.append(result)

    # Save results to JSON file
        with open("experiment_results.json", "a") as file:
            json.dump(result, file)
            file.write("\n")
