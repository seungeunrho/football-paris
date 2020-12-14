# Google Research Football Competition - liveinparis team

* The exact codes used by the team "liveinparis" at [the kaggle football competition](https://www.kaggle.com/c/google-football)

* Implementations of self-play RL from scratch with distributed actors 

* Final version of agents ranked [6th/1141](https://www.kaggle.com/c/google-football/leaderboard) (gold prize)

* You can find all the training details at [here](https://www.kaggle.com/c/google-football/discussion/201376)


## Dependencies
1. [google-research football](https://github.com/google-research/football)
2. PyTorch
3. tensorboardX
4. kaggle_environments

## Usage
```bash
python3 train.py 
# You can find args and hyper-parameters at the "arg_dict" in train.py. 
```

## training curves (vs rule base AI)
![](data/images/trained_result.png)
(x-axis : # of episodes)
1. Orange curve - vs. easy level AI
2. Blue - vs. medium level AI 

## learning system
<img src="data/images/system.PNG" height="250"></img>

Actor proceeds simulation and send rollouts(transition tuples of horizon length 30) to the central learner. Learner updates the agent with provided rollouts. Since we chose on-policy update algorithm, we used a trick to ensure perfect on-policyness(behavior policy and learning policy are equal). Actor periodically stops simulation process when the learner is updating the policy. Actor resumes simulation when it receives the newest model from the learner after training.
We used 1 actor per 1 cpu core. Our final version of agent is trained with 30 cpu cores and 1 gpu for 370 hours (cpu: AMD Ryzen Threadripper 2950X, gpu : RTX 2080). This is equivalent to 450,000 episodes, and 133M times of mini batch updates(single mini batch composed of 32 rollouts, each rollout composed of 30 state transitions).


