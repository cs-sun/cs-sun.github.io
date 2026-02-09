---
title: "Hello World: My First Blog Post"
date: 2026-01-29
description: "A brief introduction to my new blog, featuring some math and code examples."
math: true
---

Welcome to my blog! This is a simple static blog built with a custom Python script that converts Markdown to HTML.

## Why Start a Blog?

I've decided to start writing about my research interests, technical notes, and thoughts on various topics in machine learning and software engineering.

## Math Support

This blog supports KaTeX for rendering mathematical equations. Here's the famous Euler's identity:

$$e^{i\pi} + 1 = 0$$

And here's an inline example: the quadratic formula is $x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$.

For my research in graph neural networks, we often deal with message passing:

$$h_v^{(k)} = \sigma\left( W^{(k)} \cdot \text{AGG}\left( \{h_u^{(k-1)} : u \in \mathcal{N}(v)\} \right) \right)$$

## Code Examples

Here's a simple Python function:

```python
def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```

And some JavaScript:

```javascript
const greet = (name) => {
    console.log(`Hello, ${name}!`);
};
```

## What's Next?

I plan to write about:

- Graph neural network explanations
- Adversarial robustness in deep learning
- Research notes and paper summaries

Stay tuned!
