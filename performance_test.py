import time
import numpy as np
import matplotlib.pyplot as plt

class PerformanceResults:
    def __init__(self, accuracy):
        self.accuracy = accuracy
        self.inference_times = {}
    
    def add_inference_time(self, sample_size, time_taken):
        self.inference_times[sample_size] = time_taken
        
    def time_per_sample(self, sample_size):
        return self.inference_times[sample_size] / sample_size

    @property
    def true_inference_time(self):
        x = np.array(list(self.inference_times.keys()))
        y = np.array(list(self.inference_times.values()))
        m, c = np.polyfit(x, y, 1)
        return m, c

def evaluate_performance(model, test_images, test_labels):
    results = PerformanceResults(model.evaluate(test_images, test_labels)[1])
    sample_sizes = [1, 10, 100, 1000, 10000, 100000]
    for size in sample_sizes:
        num_repeats = int(np.ceil(size / len(test_images)))
        extended_test_images = np.tile(test_images, (num_repeats, 1, 1))
        sample_inputs = extended_test_images[:size]
        start_time = time.time()
        model.predict(sample_inputs, verbose=0)
        end_time = time.time()
        results.add_inference_time(size, end_time - start_time)
    return results

def render_results(results):
    print("Model Performance Evaluation:")
    print("-----------------------------")
    print(f"Accuracy: {results.accuracy * 100:.2f}%")
    print("\nInference Times:")

    max_length = max([len(str(size)) for size in results.inference_times.keys()])

    for size, time_taken in results.inference_times.items():
        print(f"For {size:<{max_length}} samples: Total time = {time_taken:.5f} seconds, Time per sample = {results.time_per_sample(size):.6f} seconds")

    print(f"\nEstimated true inference time per sample (without loading): {results.true_inference_time[0]:.6f} seconds")

    x = np.array(list(results.inference_times.keys()))
    y = np.array(list(results.inference_times.values()))
    m, c = np.polyfit(np.log(x), np.log(y), 1)

    x_dense = np.logspace(np.log10(min(x)), np.log10(max(x)), 400)
    y_dense = np.exp(np.log(x_dense)*m + c)

    plt.scatter(x, y, color='blue', label="Data")
    plt.plot(x_dense, y_dense, color='red', label=f"Fitted line")

    plt.xscale('log')
    
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.scatter(0, c, color='green')
    plt.annotate(f'Loading Time: {c:.6f}', xy=(0, c), xytext=(max(x)*0.1, c + max(y)*0.1), arrowprops=dict(facecolor='black', arrowstyle='->'))

    plt.text(max(x)*0.6, max(y)*0.5, f"True Inference Time:\n{m:.6f} sec/sample", bbox=dict(boxstyle="round", fc="white"))

    plt.title("Total Time vs. Number of Samples")
    plt.xlabel("Number of Samples")
    plt.ylabel("Total Time (seconds)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.show()


