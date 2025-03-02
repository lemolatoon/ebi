import os
import random
import sys
from tqdm import tqdm


def calculate_random_weighted_sum(seeds):
    """
    Calculate a sum of products where each product is a seed value multiplied by a random number between 0 and 5.

    Args:
    seeds (list): A list of seed values.

    Returns:
    float: The total sum of all products.
    """
    total_sum = 0

    # Generate a random number for each seed, multiply and add to the total sum
    for seed in seeds:
        t = random.randint(0, 5)  # Generate a random int number between 0 and 5
        product = seed * t
        total_sum += product
        # print(f"Seed: {seed}, Random t: {t}, Product: {product:.2f}")

    # Print the final sum
    # print(f"Total Sum: {total_sum:.2f}")
    return total_sum


# Example usage:
seeds = [1, 2, 4, 8, 16, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]


def main(output_dir: str):
    dataset_size = 2048 * 2048
    output_path = os.path.join(output_dir, f"xor_dataset_{dataset_size}.csv")
    with open(output_path, "w") as csv_file:
        for i in tqdm(range(dataset_size)):
            v = calculate_random_weighted_sum(seeds)
            v_str = f"{v:.15f}".rstrip("0").rstrip(".")
            csv_file.write(v_str)
            if i + 1 < dataset_size:
                csv_file.write(",")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python create_xor_datasets.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    main(output_dir)
