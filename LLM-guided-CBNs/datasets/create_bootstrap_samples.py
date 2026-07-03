# This is the file for creating the bootstrap samples for the different patients
import pandas as pd

# Create 10 bootstrap samples of the data
n_samples = 5
if __name__ == "__main__":
    # Read the relevant file for the data
    df = pd.read_csv("apo_data.csv")
    df = df.astype(int)

    for i in range(n_samples):
        # Create a bootstrap sample of the data
        sample = df.sample(frac=1, replace=True, random_state=i)
        sample.to_csv(f"bootstrap_sample_{i}.csv", index=False)
