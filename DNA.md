# DNA - Dynamic Neuron Activation

## Key idea

- for each layers MLP use a approximated gate projection to determine active neurons
- compute the active neurons full gate and up precision and compute SwiGLU et al
  - fill in learned bias term for silent neurons (skippable if done implicit in down projection)
- compute down projection only for active neurons on transposed matrix
  - add bias terms for silent neurons
  - use blocking and skipping

## details

### approximate gate projection

- use only sign from original gate
- use simple XOR scheme for compute


### determine active neurons

- use dead zone: smaller and larger values are considered active
- optional: use sign of gate approximation for +- bias correction


### full precision partial gate and up projection

- compute only for active neurons
- CPU friendly
- optional: compute in GPU shaders


### transposed biased down projection

- compute full residual contribution vector for each active neuron
  - stripes: pointers to N (16) columns and an N wide activation vector -> 1 residual vector
  - add vectors for all stripes (add per thread, then sum those)
- residual contribution of inactive neurons
  - depends on gate sign?
  - if no, then 
    - remove idle background from down weights -> active neurons compute the delta to inactive state
    - add an all-inactive bias residual vector


