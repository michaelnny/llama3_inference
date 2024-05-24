import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_performance(csv_filename):
    df = pd.read_csv(csv_filename)
    
    # Create a figure with a single row and two columns
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Plot Token Rate vs Concurrency in the first subplot
    df.plot(x='concurrency', y='token_rate', ax=axes[0], marker='o')
    axes[0].set_title('Token Rate vs Concurrency')
    axes[0].set_xlabel('Concurrency')
    axes[0].set_ylabel('Token Rate')

    # Plot Latency Percentiles vs Concurrency in the second subplot
    df.plot(x='concurrency', y=['p50_latency', 'p70_latency', 'p90_latency', 'p99_latency'], ax=axes[1], marker='o')
    axes[1].set_title('Latency Percentiles vs Concurrency')
    axes[1].set_xlabel('Concurrency')
    axes[1].set_ylabel('Latency (seconds)')
    axes[1].legend(['P50', 'P70', 'P90', 'P99'])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot performance metrics from a CSV file.")
    parser.add_argument('--input_csv', type=str, default='./logs/concurrency_performance_results.csv', help='Path to the CSV file.')

    args = parser.parse_args()

    plot_performance(args.input_csv)


    
