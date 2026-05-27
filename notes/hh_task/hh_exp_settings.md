# HH alignment experiment plan

Exp 1:

Goal: I want to check the mutual influence between different samples. For example, how all hiddens in the first data sample correlates to all hiddens in another data sample. I need roughly 10k values to draw my figure. So, please help me find the correct settings among the first 100 examples in GSM8K dataset. I also need the data for the following layers [0,4,8,12,20,-1].

- exp_name: GSM8K_try
- dataset_split: 'train[:100]'
- layers: [0,4,8,12,20,-1]

