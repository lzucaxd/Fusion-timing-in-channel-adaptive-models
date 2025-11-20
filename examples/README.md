# Examples and Tests

This directory contains example scripts and tests for the CHAMMI dataset implementation.

## Demos

- **`demos/demo_dataloader_no_padding.py`**: Demonstrates how DataLoader works without padding (returns list of tensors with variable channels)

- **`demos/demo_efficient_batching.py`**: Shows the efficient grouped DataLoader approach with separate DataLoaders per channel count

## Tests

- **`tests/test_chammi_dataset.py`**: Comprehensive test script that:
  - Tests basic dataset functionality
  - Tests label extraction
  - Visualizes samples from all three datasets
  - Tests DataLoader functionality
  - Saves visualizations to `./chammi_visualizations/`

## Running

```bash
# Run tests
cd tests
python test_chammi_dataset.py

# Run demos
cd demos
python demo_efficient_batching.py
```

