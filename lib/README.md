# Basic Architecture for Personalized FL
The main architecture is contained within `fl_abstract.py`. Please refer to it for more information.

## Core Learning 
 The core learning process is executed in the `next_fn()` , where users may customize the FL algorithm by modifying each component in the function.

In essence, `next_fn()` runs by doing the following:

1. A server-to-client broadcast step.
    - Denoted as `broadcast()`
2. A local client update step.
    - Denoted as `client_update()`
3. A client-to-server upload / server-side computation step.
    - Denoted as `server_compute()`
4. A server update step.
    -  Denoted as `server_update()`

Users may customize the behavior of the FL algorithm by first inheriting the `FederatedLearning` class, then overriding the respective functions inside `next_fn()`

## Dataset Partitioning

A key component to personalized FL is the method by which data is used to fine-tune and test the model.