# Importance Matrix (imatrix) Walkthrough

This document provides a detailed walkthrough of the importance matrix (imatrix) calculation in `llama.cpp`.

## What is the Importance Matrix?

The importance matrix is a concept used in model quantization. It helps to determine which weights in a neural network are more "important" for the model's performance. When quantizing a model (reducing the precision of its weights to save memory and computation), we want to be more careful with the important weights to minimize the loss of accuracy.

## How is it calculated in `llama.cpp`?

Instead of using gradients, which is a common but computationally expensive approach, `llama.cpp` uses a simplified method based on the activations.

The author of the pull request argues that to minimize the impact of quantization, one should minimize `[Sum_j (q_j - w_j) a_j]^2`, where `w_j` are the original weights, `q_j` are the quantized weights, and `a_j` are the activations.

By making some simplifying assumptions (ignoring the off-diagonal elements of the `a_i * a_j` matrix), the problem reduces to calculating the expectation value of the squared activations `<a_i^2>`. This is what `imatrix.cpp` does.

## The Formula

The importance matrix value for a given weight is proportional to the sum of the squares of the activations that are multiplied by that weight, averaged over a training dataset.

So, if `a_i` are the activations, the importance value is proportional to `Sum(a_i^2) / N`, where N is the number of tokens in the training data. The `imatrix.cpp` code calculates the sum of squares and the count separately and then divides them to get the average.

## Is the value proportional to the magnitude of the activations?

Yes, the importance value is directly proportional to the squared magnitude (or squared value) of the activations. Higher magnitude activations will result in a higher importance value for the corresponding weights.

## Code Walkthrough

Here is a walkthrough of the `imatrix.cpp` code to explain how the importance matrix is calculated in practice.

### 1. `main()` function

The `main` function is the entry point of the program. It handles the following tasks:

-   **Argument Parsing**: It parses command-line arguments using `common_params_parse`. This allows you to specify the model file, training data, output file, and other parameters.
-   **Collector Initialization**: It creates an `IMatrixCollector` object named `g_collector` and sets its parameters.
-   **Loading Previous Data**: If you provide pre-computed importance matrices using the `--in-file` argument, it loads them using `g_collector.load_imatrix()`. This is useful for combining results from multiple runs.
-   **Model and Context Initialization**: It initializes the LLaMA model and context using `common_init_from_params`.
-   **Callback Setup**: It sets up a callback function `ik_collect_imatrix` which will be called by the `ggml` backend for each node in the computation graph. This is the key to intercepting the activations.
-   **Computation**: It calls `compute_imatrix` to process the training data and collect the importance matrix data.
-   **Saving the Matrix**: Finally, it saves the computed importance matrix to a file using `g_collector.save_imatrix()`.

### 2. `compute_imatrix()` function

This function orchestrates the process of computing the importance matrix over the training data.

-   **Tokenization**: It tokenizes the input training data file.
-   **Chunking**: It processes the data in chunks of `n_ctx` tokens.
-   **Inference**: For each chunk, it performs inference using `llama_decode`. During this process, the `ggml` backend calls the `ik_collect_imatrix` callback for each operation in the computation graph.
-   **Perplexity Calculation**: It can also compute the perplexity of the model on the training data, which is a measure of how well the model predicts the data.

### 3. `IMatrixCollector::collect_imatrix()` function

This is where the core logic of collecting the importance matrix data resides.

-   **Filtering Operations**: The function is called for every operation in the computation graph, but it's only interested in matrix multiplication operations, specifically `GGML_OP_MUL_MAT` and `GGML_OP_MUL_MAT_ID`. These are the operations where weights are multiplied by activations.
-   **Getting Activations**: It gets the `src1` tensor from the `ggml_tensor`, which contains the activations.
-   **Squaring and Accumulating**: It iterates over the elements of the activation tensor, squares them, and adds the result to the `values` vector in the `m_stats` map. The `m_stats` map stores the sum of squares and the count for each tensor, with the tensor name as the key.
-   **Handling GPU Data**: If the activations are on the GPU, it copies them to the CPU before processing.

Here's a snippet from the code that shows the accumulation:

```cpp
// inside IMatrixCollector::collect_imatrix
// ...
for (int row = 0; row < (int)src1->ne[1]; ++row) {
    const float * x = (const float *) (data + row * src1->nb[1]);
    for (int j = 0; j < (int)src1->ne[0]; ++j) {
        e.values[j] += x[j]*x[j];
        e.counts[j]++;
        if (!std::isfinite(e.values[j])) {
            LOG_ERR("%f detected in %s\n", e.values[j], wname.c_str());
            exit(1);
        }
    }
}
```

### 4. `IMatrixCollector::save_imatrix()` function

This function saves the collected data to a file.

-   **Calculating the Average**: Before saving, it calculates the average of the squared activations by dividing the sum of squares (`stat.values[i]`) by the number of times the activation was used (`stat.counts[i]`). It then multiplies this by the total number of calls (`stat.ncall`) to get a value that is proportional to the sum of squares.
-   **Writing to File**: It writes the number of entries, and for each entry, it writes the tensor name, the number of calls, the number of values, and the computed importance values.

This walkthrough should give you a good understanding of how the importance matrix is calculated in `llama.cpp`. It's a clever and efficient way to determine the importance of weights for quantization without the need for a full training pipeline.
