import env as fenv
import pg_agent
import buffer
import actorcritic

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
import torch 
import time
import gym

sns.set(style='darkgrid')
sns.set(rc={'figure.figsize':(15,8)})


dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d')

factors_returns = pd.read_csv('factors_returns.csv', index_col=0, 
                    parse_dates=True, date_parser=dateparse)

strategy_returns = pd.read_csv('strategy_returns.csv', index_col=0, 
                    parse_dates=True, date_parser=dateparse)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def generate_samples(env):
    # THIS IS NOT WORKING, LEFT HERE FOR FUTURE PURPOSES
    num_episodes = 50
    weights_pred = []
    for i_episode in range(num_episodes):
        #Initialize the environment and state
        obs = env.reset()

        state = torch.tensor(factors_returns.iloc[i_episode,:],device=device)
        for t in [0]:
            # Select and perform an action
            #print(state)
            action = select_action(state)
            weights_pred.append(action)
            #weights_tensor = torch.stack(weights_pred)
            weights_tensor = action
            #print(weights_tensor.shape)
            weights_df = pd.DataFrame(index=factors_returns[i_episode:i_episode+1].index, columns = factors_returns.columns)
            weights_df.iloc[0] = weights_tensor.cpu()
            o_reward = reward(weights_df, factors_returns[i_episode:i_episode+1], strategy_returns[i_episode:i_episode+1])
            print(o_reward)
            o_reward = torch.tensor([o_reward], device=device)
            

            # Observe new state
            #last_screen = current_screen
            #current_screen = get_screen()
            if i_episode < len(factors_returns):
                #next_state = current_screen - last_screen
                next_state = torch.tensor(factors_returns.iloc[i_episode + 1,:])
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')

def buildPrices(dataframe,base):
    df= pd.DataFrame()
    for name in dataframe.columns:
        df[name+' price']= base * (1 + dataframe[name]).cumprod()
    return df

def plot_results(weights):
    df = buildPrices(factors_returns,100)
    df['Strategy price']= 100 * (1 + strategy_returns['Last Price']).cumprod()
    pred_returns = pd.DataFrame()
    pred_returns['Last Price'] = (1 + (weights * factors_returns).sum(axis=1)).cumprod().pct_change().fillna(0)
    df['Agent price']= 100 * (1 + pred_returns['Last Price']).cumprod()

    plt.plot(df[['Strategy price','Agent price']])
    plt.legend(df[['Strategy price','Agent price']].columns,loc='upper left')
    plt.show()
    return(df)


def reward(weights, factors_returns, strategy_returns):
    '''
    The assumed formula is : 
    reward = [return by agent - "groundtruth" or financial strategy] + stability of the agent (?)
    '''
    pred_returns = (1 + (weights * factors_returns).sum(axis=1)).cumprod(
        ).pct_change().fillna(0)
    tracking_error =  (pred_returns.values - strategy_returns.iloc[:,0].values
        ) * np.sqrt(250) * np.sqrt(weights.shape[1]+1)
    #turn_over = 0.0020 * 365 * ((weights - weights.shift(1)).abs().fillna(0).values
    #    ) / ((weights.index[-1] -weights.index[0]).days) * np.sqrt(
    #    weights.shape[0] * (weights.shape[1]+1)) 
    #error_terms = np.concatenate([tracking_error, turn_over.flatten()], axis=0)
    #reward = -np.sqrt(np.mean(error_terms**2))
    reward = -np.sqrt(np.mean(tracking_error**2))
    return reward




def create_submission(weights):
    # read factors
    weights = pd.DataFrame(index=factors_returns.index, columns = factors_returns.columns, data=weights)
    weights += np.random.normal(0, 0.04/ np.sqrt(250), weights.shape)
    weight_abs_diff = (weights - weights.shift(1)).abs().fillna(0)
    weight_abs_diff*= 0.0025 * 365 / ((weights.index[-1] - weights.index[0]).days) * np.sqrt(
        weights.shape[0] * (weights.shape[1]+1))
    pred_returns = (1 + (weights * factors_returns).sum(axis=1)).cumprod(
        ).pct_change().fillna(0)
    pred_returns*= np.sqrt(250) * np.sqrt(weights.shape[1]+1)
                    
    # format submission
    submission = pred_returns.to_frame() 
    submission.columns = ['Expected']
    submission.index = ['{:%Y-%m-%d}'.format(x) + '_returns' for x in pred_returns.index]
    
    for col in weight_abs_diff.columns:
        tmp = weight_abs_diff[col].to_frame()
        tmp.columns = ['Expected']
        tmp.index = [ '{:%Y-%m-%d}'.format(x) + '_{:}'.format(col) for x in tmp.index]
        submission = pd.concat([submission, tmp], axis=0)
    submission.index.names = ['Id']
    return submission




if __name__ == '__main__':
    timestr = time.strftime("%Y%m%d-%H%M%S")
    namefile = "models/ac_agent_car" + timestr
    #env = fenv.Decode_v1(factors_returns, strategy_returns)
    env = gym.make('Pendulum-v1')
    obs_space = env.observation_space.shape[0]
    act_space = env.action_space.shape[0]

    lr = [0.1,0.0001]
    batch_size = 32
    policy_epochs = 10000

    policy = actorcritic.ActorCritic(obs_space,act_space,lr,batch_size,policy_epochs)
    policy.actor.model_file = namefile
    policy.train(env)

    policy.actor.save()
    #------------Loading---------------
    #policy.actor.load("pg_agent2.pth")

    #----------- Testing --------------
    if False:
        weights = policy.test_policy(env)
        print(weights)
        plot_results(weights)

        #------------ Submission ----------
        
        subdf = create_submission(weights)
        subdf.to_csv('submission/submission'+timestr +'.csv')
    print("done")