# Gradient Descent From Scratch: Derivation, Intuition & Implementation

> A self-contained walkthrough of gradient descent for linear regression — the math behind the update rule, how to tune the learning rate, and a NumPy implementation you can run end-to-end.
>
> Companion to [`README.md`](./README.md) — see **§2.1 Mean Squared Error** for the loss function reference and **§12.1–12.5 Optimization Methods** for how this generalizes to momentum, Adam, and learning rate schedules.

---

## Table of Contents

1. [The Set-Up: Linear Regression & Loss Function](#1-the-set-up-linear-regression--loss-function)
2. [The Derivation: Finding the Gradient](#2-the-derivation-finding-the-gradient)
3. [The Update Rule](#3-the-update-rule-new-value-from-old-value)
4. [Tuning the Step: Overshooting vs. Undershooting](#4-tuning-the-step-overshooting-vs-undershooting)
5. [Implementation: Gradient Descent in NumPy](#5-implementation-gradient-descent-in-numpy)
6. [Reading the Output](#6-reading-the-output)

---

## 1. The Set-Up: Linear Regression & Loss Function

For a simple linear regression with a single feature $x$, our prediction model is:

$$f(x) = wx + b$$

Where $w$ is the weight (slope) and $b$ is the bias (y-intercept). To measure how wrong our predictions are across $m$ training examples, we use the Mean Squared Error (MSE) loss function:

$$J(w, b) = \frac{1}{2m} \sum_{i=1}^{m} \left( f(x^{(i)}) - y^{(i)} \right)^2$$

**Why the $\frac{1}{2}$?** It's a mathematical convenience. When we differentiate the squared term, the exponent $2$ brings down a multiplier that neatly cancels out this $2$, leaving a cleaner derivative (see §2 below).

This is the single-feature special case of the general MSE loss in [§2.1](./README.md#21-mean-squared-error-mse--l2-loss) of the main reference.

---

## 2. The Derivation: Finding the Gradient

To find which direction leads "downhill," we take the partial derivative of $J(w, b)$ with respect to both parameters.

**Gradient for $w$**, via the chain rule:

$$\frac{\partial J}{\partial w} = \frac{\partial}{\partial w} \left[ \frac{1}{2m} \sum_{i=1}^{m} \left( (wx^{(i)} + b) - y^{(i)} \right)^2 \right]$$

$$\frac{\partial J}{\partial w} = \frac{1}{2m} \sum_{i=1}^{m} 2\left( (wx^{(i)} + b) - y^{(i)} \right) \cdot \frac{\partial}{\partial w}(wx^{(i)} + b)$$

Since $\frac{\partial}{\partial w}(wx^{(i)} + b) = x^{(i)}$, and the $2$ cancels with the $\frac{1}{2}$:

$$\frac{\partial J}{\partial w} = \frac{1}{m} \sum_{i=1}^{m} \left( f(x^{(i)}) - y^{(i)} \right) x^{(i)}$$

**Gradient for $b$**, using the same chain rule steps (where $\frac{\partial}{\partial b}(wx^{(i)} + b) = 1$):

$$\frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \left( f(x^{(i)}) - y^{(i)} \right)$$

---

## 3. The Update Rule: New Value from Old Value

The gradient points in the direction of steepest *ascent*. Since we want to minimize the loss, we subtract the gradient to move downhill, scaled by a **learning rate** $\alpha$.

Every iteration, we simultaneously update both parameters:

$$w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial J}{\partial w}$$

$$b_{\text{new}} = b_{\text{old}} - \alpha \cdot \frac{\partial J}{\partial b}$$

This is the single-parameter instance of the general update rule in [§12.1 Gradient Descent Variants](./README.md#121-gradient-descent-variants).

---

## 4. Tuning the Step: Overshooting vs. Undershooting

The learning rate $\alpha$ is a crucial hyperparameter. Getting it wrong leads to two distinct failure modes.

### Undershooting (learning rate too small)

- **The problem**: steps are tiny, so it takes thousands of iterations to reach the minimum, wasting compute.
- **The fix**: if the loss is decreasing painfully slowly and steadily, increase $\alpha$ by a factor of 10 (e.g., $0.001 \to 0.01 \to 0.1$).

### Overshooting (learning rate too large)

- **The problem**: the step size is so large that you jump past the minimum to the opposite side of the curve. Because the slope is steeper there, the next step is even larger — the algorithm diverges (loss grows without bound).
- **The fix**: if the loss is increasing or wildly bouncing up and down, immediately cut $\alpha$ by a factor of 10.

### How to prevent both automatically

In practice, a single static $\alpha$ is rarely used. Three common strategies (each already cataloged in the main reference):

1. **Learning rate decay** — start large to cover ground quickly, then shrink over time. See [§12.5 Learning Rate Schedules](./README.md#125-learning-rate-schedules) (step decay, cosine annealing, warmup).
2. **Adaptive optimizers** — Adam or RMSprop compute a separate, adaptive rate per parameter. See [§12.3 Adaptive Learning Rate Methods](./README.md#123-adaptive-learning-rate-methods).
3. **Feature scaling** — if input features have wildly different scales (e.g., house size in sq ft vs. number of bedrooms), the loss landscape becomes a stretched, oval-shaped bowl, and gradient descent bounces back and forth uselessly. Z-score normalization turns the oval back into a circle, making $\alpha$ far easier to tune.

---

## 5. Implementation: Gradient Descent in NumPy

The code below applies exactly the derivation above to a synthetic dataset generated from a known ground truth ($w=2$, $b=5$), so you can check convergence directly against the target values.

```python
import numpy as np

# 1. Generate synthetic linear data (y = 2x + 5 + some noise)
np.random.seed(42)
m = 100  # Number of data points
X = 2 * np.random.rand(m, 1)
y = 5 + 2 * X + np.random.randn(m, 1) * 0.1  # True w = 2, b = 5

# 2. Gradient Descent Function
def gradient_descent(X, y, learning_rate=0.1, iterations=1000):
    m = len(y)

    # Initialize parameters randomly
    w = np.random.randn(1, 1)
    b = np.zeros((1, 1))

    # Track loss history to monitor learning path
    loss_history = []

    for i in range(iterations):
        # Forward pass: compute predictions
        y_pred = np.dot(X, w) + b

        # Calculate Mean Squared Error (MSE) Loss
        loss = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        loss_history.append(loss)

        # Backward pass: compute gradients (the derivations we did earlier!)
        dw = (1 / m) * np.dot(X.T, (y_pred - y))
        db = (1 / m) * np.sum(y_pred - y)

        # Update rule: New = Old - Learning_Rate * Gradient
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Print progress every 100 steps
        if i % 100 == 0:
            print(f"Iteration {i:4d} | Loss: {loss:.6f} | w: {w[0,0]:.4f} | b: {b[0,0]:.4f}")

    return w, b, loss_history

# 3. Run the optimization
learning_rate = 0.1
final_w, final_b, history = gradient_descent(X, y, learning_rate, iterations=500)

print("\n--- Training Complete ---")
print(f"Target Values -> w: 2.0000, b: 5.0000")
print(f"Learned Values -> w: {final_w[0,0]:.4f}, b: {final_b[0,0]:.4f}")
```

### Mapping the code back to the math

| Code | Math |
| --- | --- |
| `y_pred = np.dot(X, w) + b` | $f(x^{(i)}) = wx^{(i)} + b$ |
| `loss = (1/(2*m)) * np.sum((y_pred - y)**2)` | $J(w,b) = \frac{1}{2m}\sum_i (f(x^{(i)}) - y^{(i)})^2$ |
| `dw = (1/m) * np.dot(X.T, (y_pred - y))` | $\frac{\partial J}{\partial w} = \frac{1}{m}\sum_i (f(x^{(i)}) - y^{(i)}) x^{(i)}$ |
| `db = (1/m) * np.sum(y_pred - y)` | $\frac{\partial J}{\partial b} = \frac{1}{m}\sum_i (f(x^{(i)}) - y^{(i)})$ |
| `w = w - learning_rate * dw` | $w_{\text{new}} = w_{\text{old}} - \alpha \cdot \frac{\partial J}{\partial w}$ |
| `b = b - learning_rate * db` | $b_{\text{new}} = b_{\text{old}} - \alpha \cdot \frac{\partial J}{\partial b}$ |

This is **batch gradient descent** — every iteration uses the full dataset (`X`, `y` in one shot via matrix multiplication) rather than a single sample (SGD) or a mini-batch. See [§12.1](./README.md#121-gradient-descent-variants) for how the update rule changes for those variants.

---

## 6. Reading the Output

With `learning_rate = 0.1` and 500 iterations, the loss should decrease monotonically toward zero and `final_w`/`final_b` should converge close to the true generating values (`w=2.0`, `b=5.0`) — a smooth "good $\alpha$" trajectory like the middle panel in [§V3.2 Learning Rate Effect](./README.md#v32-learning-rate-effect).

Two quick experiments worth trying, tying straight back into §4:

- Set `learning_rate = 0.001` and watch the loss decrease but converge far too slowly within the same 500 iterations (undershooting).
- Set `learning_rate = 5` (or higher) and watch the printed loss grow instead of shrink, sometimes diverging to very large numbers or `nan` (overshooting).
