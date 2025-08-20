# README.md
# Variational Autoencoders for Well Log Facies Classification

This project implements a Variational Autoencoder (VAE) for unsupervised facies classification using well log data. The workflow includes data loading, preprocessing, model training, clustering in latent space, and visualization of results.

## Features

- Reads and merges raw and processed LAS well log files
- Preprocesses and standardizes tabular data
- Defines a PyTorch-based VAE for dimensionality reduction and feature learning
- Clusters latent representations using KMeans
- Visualizes latent space and well logs with facies predictions
- Computes and displays classification metrics

## Usage

1. Place your raw and processed LAS files in the appropriate directory.
2. Update the `raw_input` and `processed_input` paths in [`main.py`](main.py).
3. Run the script:

```sh
python main.py
```

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list of dependencies.

## Notes

- Ensure your LAS files contain the required columns as referenced in the code.
- The script is set up for GPU acceleration if available.

## License

MIT License