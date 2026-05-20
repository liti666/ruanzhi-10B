"""
Run this script once to pre-download and cache the IMDB dataset locally.
Subsequent runs will use the cache and work offline.
"""
import nltk
from datasets import load_dataset

def main():
    print("Downloading IMDB dataset...")
    dataset = load_dataset("imdb")
    print(f"  Train: {len(dataset['train'])} examples")
    print(f"  Test:  {len(dataset['test'])} examples")
    print("  Done.")

    print("\nDownloading NLTK resources required by TextAttack...")
    for pkg in ["punkt", "averaged_perceptron_tagger", "omw-1.4", "wordnet"]:
        nltk.download(pkg, quiet=True)
    print("  Done.")

    print("\nAll data ready. You can now run the training and attack scripts.")

if __name__ == "__main__":
    main()
