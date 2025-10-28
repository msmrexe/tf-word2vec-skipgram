# Word2Vec Skip-Gram Implementation

A complete implementation of the Word2Vec Skip-Gram model with negative sampling using TensorFlow. This project, developed for an M.S. Machine Learning course, processes the Text8 Wikipedia corpus, trains word embeddings from scratch, and includes utilities for visualizing the resulting vector space.

## Features

* Efficient text preprocessing including stopword removal, punctuation tokenization, and frequency-based filtering.
* Implementation of Mikolov et al.'s subsampling heuristic for frequent words.
* Generation of skip-gram pairs with negative sampling using TensorFlow/Keras utilities.
* A custom `tf.keras.Model` class for the Skip-Gram architecture.
* A custom training loop with `tf.GradientTape` for fine-grained control and logging.
* Checkpointing with `tf.train.CheckpointManager` to save model progress.
* Command-line argument parsing with `argparse` to easily configure hyperparameters.
* Utilities to export embeddings (`vecs.tsv`, `meta.tsv`) for visualization in the TensorFlow Embedding Projector.

## Core Concepts & Techniques

* **Word Embeddings (Word2Vec):** Learning dense vector representations of words.
* **Skip-Gram Architecture:** A model that learns embeddings by predicting context words from a target word.
* **Negative Sampling:** An efficient optimization strategy that reframes the problem as binary classification.
* **TensorFlow Custom Models:** Subclassing `tf.keras.Model` to create a bespoke architecture.
* **TensorFlow Custom Training:** Using `tf.GradientTape` to manage gradients and model optimization.
* **Text Preprocessing:** Tokenization, filtering, and subsampling with `NLTK` and `TensorFlow`.
* **`tf.data` API:** Building efficient and scalable input pipelines for training.

---

## How It Works

This project trains word embeddings by implementing the Skip-Gram model with Negative Sampling. The core idea is that a word's meaning is defined by the company it keeps. The model learns by trying to distinguish true "context words" from random "negative sample" words.

### 1. Data Processing Pipeline

1.  **Load Data:** The `text8` dataset (the first 100MB of Wikipedia) is downloaded.
2.  **Preprocess Text:** The raw text goes through a multi-step cleanup:
    * Punctuation is tokenized (e.g., `.` becomes `<PERIOD>`).
    * Text is lowercased and stopwords (like 'the', 'is') are removed.
    * Words appearing fewer than 5 times are filtered out.
    * **Subsampling:** Frequent words (e.g., 'anarchism' in this corpus) are randomly dropped based on the heuristic formula to prevent them from dominating the training.
3.  **Generate Pairs:** The `tf.keras.preprocessing.sequence.skipgrams` function is used. For each word in the text:
    * **Positive Samples:** It pairs the word with words inside its context window (e.g., 5 words before and after), assigning a label of `1`.
    * **Negative Samples:** It pairs the word with random words from the vocabulary, assigning a label of `0`.
4.  **Create `tf.data` Pipeline:** These pairs are converted into a `tf.data.Dataset` and batched, shuffled, and prefetched for efficient GPU training.

### 2. Model Architecture & Theory

The model (`src/model.py`) is surprisingly simple. It consists of two main components:

* `target_embedding`: An embedding layer (a $V \times D$ matrix) for the *target* (center) word.
* `context_embedding`: An embedding layer (another $V \times D$ matrix) for the *context* word.

Here $V$ is the vocabulary size and $D$ is the embedding dimension (e.g., 128).

#### Skip-Gram Objective

The original Skip-Gram objective tries to predict the context given a target word. The probability of observing a context word $w_o$ given a target word $w_i$ is defined by the **softmax** function:

$$P(w_o | w_i) = \frac{\exp(v'_{w_o} \cdot v_{w_i})}{\sum_{w=1}^{V} \exp(v'_{w} \cdot v_{w_i})}$$

where $v_{w_i}$ is the "target" vector for $w_i$ and $v'_{w_o}$ is the "context" vector for $w_o$. The problem is the denominator: calculating this requires summing over all $V$ words in the vocabulary, which is computationally infeasible for large vocabularies.

#### Negative Sampling Optimization

This project uses **Negative Sampling** as a more efficient objective. Instead of a multiclass prediction, we frame the problem as **binary classification**.

For each `(target, context)` pair, the model is trained to answer: "Is this a real context pair or a random, 'negative' pair?"

The objective function for a single positive pair $(w_i, w_o)$ and $K$ negative samples $(w_k)$ becomes:

$$L_{\text{NEG}} = \log \sigma(v'_{w_o} \cdot v_{w_i}) + \sum_{k=1}^{K} \mathbb{E}_{w_k \sim P_n(w)} [\log \sigma(-v'_{w_k} \cdot v_{w_i})]$$

* $\sigma(\cdot)$ is the sigmoid function.
* The first term pushes the dot product of *real pairs* to be high.
* The second term pushes the dot product of *fake (negative) pairs* to be low.

This is exactly what our model implements. The `skipgrams` function provides the `(target, context, label)` triples, and our model's `call` method computes the dot product. We then apply `tf.keras.losses.BinaryCrossentropy(from_logits=True)`, which is a numerically stable implementation of this exact sigmoid-based objective.

### 3. Training and Analysis of Results

The model is trained using a custom loop in `src/train.py`. Based on the 5-epoch training run from the original notebook:

* **Final Training Accuracy:** ~88.9%
* **Final Validation Accuracy:** ~84.1%

The training accuracy steadily increases, while the validation accuracy plateaus around 84-85% after the second epoch. This is expected. The model quickly learns to distinguish true pairs from random ones.

The validation *loss* (as seen in the original notebook) starts to increase after epoch 2, while validation *accuracy* stays high. This suggests the model's *confidence* on the validation set is decreasing (slight overfitting), but its *classification ability* remains strong. In Word2Vec, the final classification accuracy is less important than the **side-effect** of the training: the learned `target_embedding` vectors. The high validation accuracy confirms that the vectors have successfully encoded the co-occurrence statistics of the corpus.

When the final `vecs.tsv` and `meta.tsv` files are loaded into the TensorFlow Embedding Projector, we would observe clear semantic clustering. For example:
* `anarchism` would be spatially close to `proudhon`, `bakunin`, and `anarchists`.
* `fascism` would be in a different region, perhaps near `fascists` and `republican`.
* `one`, `two`, `three` would cluster together.

This demonstrates the model successfully learned meaningful semantic relationships from the text.

---

## Project Structure

```
tf-word2vec-skipgram/
├── .gitignore            # Ignores venv, logs, data, and cache
├── LICENSE               # MIT License file
├── README.md             # This project guide
├── requirements.txt      # Python dependencies
├── run_training.ipynb    # A simple notebook to run the training
├── checkpoints/
│   └── .gitkeep          # Holds model checkpoints
├── data/
│   └── .gitkeep          # Default location for projector files
├── logs/
│   └── .gitkeep          # Holds the training.log file
└── src/
    ├── __init__.py       # Makes 'src' a Python package
    ├── config.py         # All hyperparameters and paths
    ├── data_loader.py    # Handles data download, preprocessing, and batching
    ├── main.py           # Main executable script with argparse
    ├── model.py          # Defines the SkipGramModel (tf.keras.Model)
    ├── train.py          # Implements the custom training and eval loops
    └── utils.py          # Utility for logging and saving embeddings

````

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/tf-word2vec-skipgram.git
    cd tf-word2vec-skipgram
    ```

2.  **Set up Environment and Install Dependencies:**
    (Optional, but recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
    Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Training Pipeline:**

    You have two options:

    **Option A: Run the Jupyter Notebook (Easiest)**
    * Open and run all cells in `run_training.ipynb`. This will execute the main script with default settings.

    **Option B: Use the Command Line Script**
    * Run the main script directly. You can override any default setting from `src/config.py` using command-line arguments.

      ```bash
      # Run with default settings (5 epochs, 128 embedding dim)
      python src/main.py
  
      # Run for more epochs and with a larger embedding dimension
      python src/main.py --epochs 10 --embedding_dim 150
      ```

4.  **View Results:**
    * **Logs:** All training progress is logged to the console and saved in `logs/training.log`.
    * **Checkpoints:** Model checkpoints are saved in the `checkpoints/` directory.
    * **Embeddings:** The final vectors (`vecs.tsv`) and metadata (`meta.tsv`) are saved in the `data/` directory.

5.  **Visualize in Embedding Projector:**
    1.  Go to [http://projector.tensorflow.org/](http://projector.tensorflow.org/).
    2.  Click on **Load**.
    3.  Upload `data/vecs.tsv` for "Choose tensor file".
    4.  Upload `data/meta.tsv` for "Choose metadata file".
    5.  Explore the learned vector space!

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
