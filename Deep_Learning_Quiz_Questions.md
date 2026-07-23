# Deep Learning Quiz Questions & Solutions

This document consolidates all the questions, explanations, and corresponding diagrams covered in our deep learning discussion.

---

## Question 1: Forward & Backward Propagation Cache

**Question:**  
Which of the following is stored in the 'cache' during forward propagation for later use in backward propagation?
- $W^{[l]}$
- $Z^{[l]}$
- $b^{[l]}$

**Answer:**  
**$Z^{[l]}$** (along with $A^{[l-1]}$, $W^{[l]}$, and $b^{[l]}$)

**Explanation:**  
During forward propagation, the linear activation $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$ is saved in the cache. During backpropagation, calculating $dZ^{[l]} = dA^{[l]} * g'^{[l]}(Z^{[l]})$ requires $Z^{[l]}$ to evaluate the derivative $g'^{[l]}$ of the activation function.

---

## Question 2: Hyperparameters during Backpropagation

**Question:**  
During the backpropagation process, we use gradient descent to change the hyperparameters. True/False?

**Answer:**  
**False**

**Explanation:**  
Gradient descent updates the **parameters** of the model (weights $W$ and biases $b$). **Hyperparameters** (such as learning rate $lpha$, number of layers, hidden unit counts) are set before training and remain fixed during backpropagation.

---

## Question 3: Feature Representations in Early Layers

**Question:**  
Which of the following images is more likely related to the early layers of a deep neural network? Is the third image where we have noise?

| Image 1 (Middle Layers) | Image 2 (Later Layers) | Image 3 (Early Layers) |
|:---:|:---:|:---:|
| ![Middle Layer Features](https://i.imgur.com/image1_placeholder.png) | ![Later Layer Features](https://i.imgur.com/image2_placeholder.png) | ![Early Layer Features](https://i.imgur.com/image3_placeholder.png) |

**Answer:**  
The **Third Image** represents the **early layers**.

**Explanation:**  
- **Early Layers (Image 3):** Detect low-level primitives like edges, lines, and gradients at various orientations. These are **not noise**; they are Gabor-like directional edge detectors.
- **Middle Layers (Image 1):** Combine edges into facial features (eyes, noses, lips).
- **Later Layers (Image 2):** Combine features into complete faces.

---

## Question 4: Vectorization Across Examples

**Question:**  
Vectorization allows us to compute $A^{[l]}$ for all the examples on a batch at the same time without using a for loop. True/False?

**Answer:**  
**True**

**Explanation:**  
By stacking example vectors column-wise into matrices $X = [x^{(1)}\ x^{(2)}\ \dots\ x^{(m)}]$, matrix operations process the entire mini-batch simultaneously.

---

## Question 5: Parameter Initialization Loop

**Question:**  
Assume we store the values for $n^{[l]}$ in an array called `layer_dims = [n_x, 4, 3, 2, 1]`. Which for-loop initializes the parameters correctly?

**Answer:**  
```python
for i in range(len(layer_dims) - 1):
    parameter['W' + str(i+1)] = np.random.randn(layer_dims[i+1], layer_dims[i]) * 0.01
    parameter['b' + str(i+1)] = np.random.randn(layer_dims[i+1], 1) * 0.01
```

**Explanation:**  
- $W^{[l]}$ has shape $(n^{[l]}, n^{[l-1]})$, which corresponds to `(layer_dims[i+1], layer_dims[i])`.
- $b^{[l]}$ has shape $(n^{[l]}, 1)$, which corresponds to `(layer_dims[i+1], 1)`.

---

## Question 6: Layer Sizes (Network Diagram 1)

**Question:**  
What are all the values of $n^{[0]}, n^{[1]}, n^{[2]}, n^{[3]}, n^{[4]}$ for the network shown below?

```
Input (x1..x4) -> Layer 1 (4 units) -> Layer 2 (3 units) -> Layer 3 (2 units) -> Layer 4 (1 unit)
```

![Network Diagram 1](https://i.imgur.com/neural_network_1.png)

**Answer:**  
**4, 4, 3, 2, 1**

**Explanation:**  
- $n^{[0]} = 4$ (Input features $x_1, x_2, x_3, x_4$)
- $n^{[1]} = 4$ hidden units
- $n^{[2]} = 3$ hidden units
- $n^{[3]} = 2$ hidden units
- $n^{[4]} = 1$ output unit ($\hat{y}$)

---

## Question 7: Gradient Calculation of $dA^{[l]}$

**Question:**  
During forward propagation, to calculate $A^{[l]}$, you use $g^{[l]}(Z^{[l]})$. True/False: During backward propagation, you calculate $dA^{[l]}$ from $Z^{[l]}$.

**Answer:**  
**False**

**Explanation:**  
$dA^{[l]}$ is computed from the subsequent layer $l+1$ using $(W^{[l+1]})^T dZ^{[l+1]}$, while $dZ^{[l]}$ is calculated using $dA^{[l]} * g'^{[l]}(Z^{[l]})$.

---

## Question 8: Deep vs. Shallow Networks

**Question:**  
For any mathematical function you can compute with an $L$-layered deep neural network with $N$ hidden units, there is a shallow neural network that requires only $\log N$ units, but it is very difficult to train. True/False?

**Answer:**  
**False**

**Explanation:**  
To represent complex functions learned by deep networks, shallow networks require **exponentially more** units ($O(2^N)$ or $O(2^L)$), not fewer ($\log N$).

---

## Question 9: Dimension Shapes (Network Diagram 2)

**Question:**  
Given a network with input size 4, Layer 1 size 3, Layer 2 size 4, and Layer 3 size 1:

![Network Diagram 2](https://i.imgur.com/n1_diagram.png)

Which statements are true?
- $W^{[1]}$ shape: **(3, 4)**
- $b^{[1]}$ shape: **(3, 1)**
- $W^{[2]}$ shape: **(4, 3)**

---

## Question 10: Dimension Shapes (Network Diagram 3)

**Question:**  
Given a network with input size 4, Layer 1 size 4, Layer 2 size 3, and Layer 3 size 1:

![Network Diagram 3](https://i.imgur.com/f60e_diagram.png)

Which statements are true?
- $W^{[1]}$ shape: **(4, 4)**
- $b^{[1]}$ shape: **(4, 1)**
- $W^{[2]}$ shape: **(3, 4)**
- $b^{[2]}$ shape: **(3, 1)**
- $W^{[3]}$ shape: **(1, 3)**
- $b^{[3]}$ shape: **(1, 1)**

---

## Question 11: General Dimension of Bias Vector $b^{[l]}$

**Question:**  
In the general case, what is the dimension of $b^{[l]}$, the bias vector associated with layer $l$?

**Answer:**  
**$b^{[l]}$ has shape $(n^{[l]}, 1)$**

---

## Question 12: Purpose of the Cache

**Question:**  
We use the "cache" in our implementation of forward and backward propagation to pass useful values to the next layer in forward propagation. True/False?

**Answer:**  
**False**

**Explanation:**  
Values in the cache are stored during forward propagation to be used by the **same layer during backward propagation**, not passed forward to the next layer.

---

## Question 13: Layer Counting Convention

**Question:**  
For a network with 4 hidden layers of 2 units each and 1 output unit:

![Network Diagram 4](https://i.imgur.com/9ae4_diagram.png)

How many total layers are in the network?

**Answer:**  
**5 layers** ($L = 5$: 4 hidden layers + 1 output layer). Input is Layer 0 and is not counted toward $L$.

---

## Question 14: Activation Function Awareness in Backprop

**Question:**  
During forward propagation, in the forward function for layer $l$, you need to know the activation function $g^{[l]}$. During backpropagation, the corresponding backward function also needs to know the activation function for layer $l$. True/False?

**Answer:**  
**True**

**Explanation:**  
Backpropagation computes $dZ^{[l]} = dA^{[l]} * g'^{[l]}(Z^{[l]})$. Because $g'^{[l]}$ depends directly on whether $g^{[l]}$ is ReLU, Sigmoid, Tanh, etc., the backward pass must know the activation function used.
