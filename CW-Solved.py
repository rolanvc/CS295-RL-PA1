import gym

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'CliffWalking' in env:
       print("Remove {} from registry".format(env))
       del gym.envs.registration.registry.env_specs[env]

def main():
    env = gym.make("CliffWalking:CliffWalking-v0")
    state = env.reset()
    newstate, reward, done, info = env.step(3)
    print (newstate, reward, done)
    newstate, reward, done, info = env.step(2)
    print (newstate, reward, done)
    newstate, reward, done, info = env.step(3)
    print (newstate, reward, done)


if __name__ == "__main__":
    main()