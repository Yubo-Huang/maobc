# Multi-agent offline behavior clone

This codebase is the implementation of paper named **Optimizing an offline reinforcement learning control policy for wind farms**.

1. `./data` directory contains the training samples of 8 wind turbines
2. `./learn` directory saved the learned control policy

## How to train MAOBC

First, cloning this repo to your local computer using 

```
git clone https://github.com/LuckyYubo/maobc.git

cd maobc
```

Then, running the `.sh` file:

```
./run_dual.sh
```