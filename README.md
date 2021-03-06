# TSP Solver

Pytorch implementation of
[Neural Combinatorial Optimization with Reinforcement Learning](http://arxiv.org/abs/1611.09940)  

<img src="imgs/pointer_net.png" height="200">  
The neural network consists in LSTM encoder decoder with an attention module connecting the decoder to the encoder.  
The model is trained by Policy Gradient.  

## What is Traveling Salesman Problem(TSP)
Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?  [Wikipedia](https://en.wikipedia.org/wiki/Travelling_salesman_problem)  

<img src="imgs/tsp-bad-good-case.png" height="300">   


## Prerequisites
- pytorch
- matplotlib
- [or-gym](https://github.com/hubbs5/or-gym)
```
pip install or-gym
```

This repository is tested ...

- Windows 10
- Python 3.6
- Pytorch 1.7.1


## Reward
We did not use the reward function in the paper.  
We used the sum of rewards from the emulator. but the emulator doesn't go back to the starting city.  
After the game, the distance from the last city and the starting city is added to the total reward.  

## Usage
### Active search
```
python tsp_trainer.py --mode active-search
```
### Actor critic
```
python tsp_trainer.py --mode actor-critic
```
### Test
```
python tsp_tester.py --seq_len 20 --actor_dir ./result/actor.pth
```

#### [configs](config.py)
```
--lr 3e-4
--embedding_size 128
--hidden_size 128
--grad_clip 1.5
--decay 0.01

--n_glimpses 2
--tanh_exploration 10
--beta 0.99
--episode 1000
--seq_len10

--mode active-search
--result_dir ./result

--actor_dir ./result/actor.pth  #use tsp_tester
```
## Results

### Active search
<img src="imgs/9-99_episode_result.png" height="200"> <img src="imgs/909-999_episode_result.png" height="200">  
<img src="imgs/episode_length.png" height="200"> <img src="imgs/active-searchloss.png" height="200">  

### Actor critic
<img src="imgs/actor-critic_episode_length.png" height="200"> <img src="imgs/actor-critic_loss.png" height="200">   



## Prior knowledge
- RL (Actor-critic)
- [Pointer Network](https://arxiv.org/abs/1506.03134)

