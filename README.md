To run this code and watch it in action:

Simply download from requirements.txt, navigate to rl/src, and run python3 train_headless.py. Entry point is, of course, in train_headless.



Hey, this is all that's written by Delta. 
Most of my changes are in the RL folder. If you want to see it in action, please run train_headless.py; docker is not needed. 

Right now, I'm still training the model, but it seems that a REINFORCE model is just not good enough to crack this kind of difficulty. I will work on moving to more complicated models soon. Have been ironing out bugs with the current one, especially with editing the learning rate. DO NOT CHANGE THE LEARNING RATE. thanks!

Soon, I really have to merge visualize.py with train_headless.py, since visualize has deprecated logic. However, it's still very useful for debugging, and I'm an idiot; I shouldn't have duplicated run logic in both files. Visualize logic should have drawn from train_headless.py, so if you have time, I would love if you could help with that. I'm super tired of this tbh and I'm nearly dead....

Hmmmmh. What else. I think there is some mistaken logic with the guards, and the way they store their frame_stacks. Anyway, that's all! I'm going to shower and sleep now.

Edit: This goes without saying, but do not edit checkpoints/latest.pt -- if you do, my macbook will have suffered for nothing, the poor thing.

Edit 2: If you have time, please check my logic for updating the terminal reward and truncating the reward_buffers when the game terminates. I have a strong suspicion that it isn't working properly, since the guards aren't really getting better. Additionally....... I think I might need to change gamma to 1, or reduce number of episodes per round in gridworld. Lastly, I need to add an entropy loss in reinforce.py and consequently, episode_buffer.py! If you're kind enough, you can do that for me. Just let me know beforehand, before I waste time on it myself. 