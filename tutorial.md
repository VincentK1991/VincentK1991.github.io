---
layout: page
title: Tutorials
permalink: /tutorial/
---

* TOC
{:toc}
# Introduction

This page groups a set of work that I created as a learning materials, mainly to teach myself, other  interested readers, and as a repository for me to look up at leter time. Setting apart from the project, the tutorials are meant to be pedagogical, rather than to showcase the work. Click on the topic to re-direct to the sites. The subjects that I am most interested in include **machine learning**, **deep learning**, **Bayesian inference**.

---

# [Tutorials on Bayesian Regression using Pymc3](https://vincentk1991.github.io/Bayesian-regression-tutorial/)

I gives my take on probabilistic programming as a specialized way of coding to implement statistical calculation. I briefly introduce Pymc3 a popular package in Python for probabilistic programming. I then use Pymc3 to explore various Bayesian regression techniques including *multiple regression*, *controlling for a variable*, *causal inference*, *hierarchical modeling*, *class interaction*, and a light touch on *generalized linear model*.

![Figure 1]({{ site.baseurl }}/images/pymc3tutorial/meme.jpg "online ads"){: .center-image }
<p align="center">
    <font size="4"> </font>
</p>

---

# [Dealing with Class Imbalance with Weighted Loss Function](https://github.com/VincentK1991/Deep_Learning_Misc/tree/master/imbalanced_classification)

Using the wellknown Credit Card Fraud dataset, I explore how to write a deep learning model for classification with highly imbalanced dataset. I will also discuss what appropriate measurement to use, and how to find appropriate training hyperparameters.

---

# [Tutorials on Text Classification](https://github.com/VincentK1991/Authorship_attribution/blob/master/Machine_Learning_Guide_to_Authorship_identification.ipynb)

I provid a Colab Notebook for a simple text classification problem. Anyone is invited to make a copy and play with it.
I choose authorship attribution, i.e. question of how wrote it. This is the same type of problem as sentiment analysis of text. I look at 4 machine learning approaches: *TFIDF Vectorization + Regression*, *Naive-Bayes Classification*, *Recurrent Neural Network*, and *BERT Classification*. For each approach, I discuss the rationale, as well as the pros and cons.

---

# [Tutorials on Transfer Learning](https://github.com/VincentK1991/transfer_learning/blob/master/primer_to_transfer_learning_and_encoder.ipynb)

I provide a Colab Notebook for a simple transfer learning. For illustration, I tackle an image classification problem using pre-trained neural network model (resnet). I also discuss the rationale behind transfer learning, and show line-by-line how to implement it in Pytorch.

---

# [Deploying Web Application on AWS]()

A simple tutorial on how to deploy a web application on Amazon Web Services. See the final product [here](http://streeteasy-dashboard-aws-dev.us-west-2.elasticbeanstalk.com/)

---

# [Survival Analysis](https://github.com/VincentK1991/IBM_attrition_HR/blob/master/IBM_attrition_Apr03_2019.ipynb)

This is a learning material on survival analysis. I study the employee attrition model, i.e. looking at what factors make employee leave a company. I discuss a few concepts pertaining to survival analysis, such as censoring, hazard, and lifetime. Most of the work is done using a package for survival analysis called "lifelines".
