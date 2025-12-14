# 1. separate-trainer-losses

## Status
Proposed

## Context
The initial implementation has training loop as part of the model. We want generalization for loss functions, regularization methods and training strategies

## Decision
The proposal is to make separate structures:
    - Loss function(MSE/MAE/BCE/ with or without regularization). Than will hardcode gradient formula for now.
    - Optimizer - SGD to begin with, with learning rate making update step
    - Trainer - with batch_size, training loop, stopping criterion
    - Metrics
    - Logging


## Consequences
Can use one model for various tasks(regression, classification)
Training loop can become fallible
Fitted model is free from training hyperparameters

## Alternatives Considered
Implementing real autograd - postpone for next stages.
sklearn/keras style fit-predict - too stateful and object oriented, violates DRY and can not be generalized to other model types ever.

## References
- [Link to issue, PR, discussion, etc.](https://github.com/vzaguskin/machinelearne-rs/issues/1)