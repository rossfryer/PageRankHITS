# PageRank & HITS Interactive Tutor

An interactive Streamlit application for learning and experimenting with PageRank and HITS algorithms.

## Features

- **Interactive Graph Editor**: Add/remove nodes and edges
- **Algorithm Visualization**: Step-by-step iteration tracking
- **Multiple Presets**: Star, chain, two clusters, and dangling node examples
- **Export/Import**: Save and load graphs as JSON
- **Educational Tools**: Prediction checking and convergence visualization

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run pagerankhits.py
```

## Usage

1. Choose an algorithm (PageRank or HITS)
2. Adjust parameters in the sidebar
3. Load a preset or create your own graph
4. Click "Run algorithm" to see the results
5. Use the iteration slider to step through the algorithm's progress

## Algorithms

- **PageRank**: Uses damping factor and handles dangling nodes
- **HITS**: Computes both authority and hub scores with L1 or L2 normalization



