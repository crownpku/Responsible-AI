# Responsible AI

![](imgs/responsible-ai.png)

This is a demo project of using Responsible AI technology provided by Google to build responsible machine learning applications.

## Content

### Define Problem

#### Who is my ML system for?

The way actual users experience your system is essential to assessing the true impact of its predictions, recommendations, and decisions. Make sure to get input from a diverse set of users early on in your development process.

### Construct and prepare data

#### Am I using a representative dataset?

Is your data sampled in a way that represents your users (e.g. will be used for all ages, but you only have training data from senior citizens) and the real-world setting (e.g. will be used year-round, but you only have training data from the summer)?

#### Is there real-world/human bias in my data?

Underlying biases in data can contribute to complex feedback loops that reinforce existing stereotypes.

### Build and train model

#### What methods should I use to train my model?

Use training methods that build fairness, interpretability, privacy, and security into the model.

### Evaluate model

#### How is my model performing?

Evaluate user experience in real-world scenarios across a broad spectrum of users, use cases, and contexts of use. Test and iterate in dogfood first, followed by continued testing after launch.

![](/imgs/equal_accuracy.png)

### Deploy and monitor

#### Are there complex feedback loops?

Even if everything in the overall system design is carefully crafted, ML-based models rarely operate with 100% perfection when applied to real, live data. When an issue occurs in a live product, consider whether it aligns with any existing societal disadvantages, and how it will be impacted by both short- and long-term solutions.

# Reference
https://www.tensorflow.org/resources/responsible-ai
