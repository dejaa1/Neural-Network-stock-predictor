import os
from math import pi

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from model import MyNetwork
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm

except ImportError:
    tqdm = None


def get_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


device = get_device()


def pre_process(file_path: str) -> dict:

    df = pd.read_csv(file_path)
    stocks = sorted(list(set(df['Stock'])))  # stocks in alphabetical order

    stock_dict = dict()

    for stock in stocks:
        # >>> YOUR CODE HERE
        df1 = (df[df['Stock'] == stock])
        #
        # section = df1.loc[:, ['Date']]
        #
        #
        # df1['Date'] = pd.to_datetime(section)
        df1 =  df1.sort_values(by='Date', inplace=False)
        print(df1)
        stock_dict[stock] = df1

        # <<< END YOUR CODE

    return stock_dict


def plot_data(stock_dict: dict) -> None:

    stocks = list(stock_dict.keys())
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout(pad=3.0)
    for i in range(2):
        for j in range(2):
            axs[i, j].plot(stock_dict[stocks[i * 2 + j]]['Close'].values)
            axs[i, j].set_title(f'{stocks[i * 2 + j]}')

    for ax in axs.flat:
        ax.set(xlabel='days', ylabel='close price')

    plt.savefig(os.path.join(os.path.dirname(__file__), 'stocks_history.png'))


def split_stock(stock_info: pd.DataFrame) -> tuple:

    x = []
    y = []


    num_elem = 5
    stock_list = (stock_info['Close']).to_list()
    for i in range(len(stock_list) - num_elem):
        x.append(stock_list[i:i + num_elem])
        y.append(stock_list[i + num_elem])

    splice_index = int(len(stock_list) * 0.7)

    x_train_stock, y_train_stock, x_val_stock, y_val_stock = np.asarray(x[:splice_index]), np.asarray(
        y[:splice_index]), np.asarray(x[splice_index:]), np.asarray(y[splice_index:])


    return (x_train_stock, y_train_stock, x_val_stock, y_val_stock)


def get_train_valid(stock_dict: dict) -> tuple:

    x_train, y_train, x_val, y_val = np.empty((0, 5),), np.empty((0,), ), np.empty((0, 5), ), np.empty((0,), )

    for stock in stock_dict:

        vals = split_stock(stock_dict[stock])

        x_train = np.concatenate((x_train, vals[0]), axis=0)
        y_train = np.concatenate((y_train, vals[1]))
        x_val = np.concatenate((x_val, vals[2]), axis=0)
        y_val = np.concatenate((y_val, vals[3]))

    return (x_train, y_train, x_val, y_val)


def my_NLLloss(pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:



    val = torch.log(torch.sqrt(torch.tensor(2*torch.pi))) + (0.5 * pred[:, 1]) + ((torch.pow(y-pred[:, 0], 2)) / (2*torch.exp(pred[:, 1])))
    nll_loss = torch.sum(val)


    return nll_loss


def train(data: tuple, max_epochs: int = 200, seed=12345) -> tuple:



    torch.set_grad_enabled(True)
    torch.set_default_dtype(torch.float64)
    learning_rate = 0.0006
    ##prev = 0.0005
    torch.manual_seed(seed)

    if tqdm is not None:
        iterator = tqdm(range(max_epochs))
    else:
        iterator = range(max_epochs)



    net = MyNetwork(5, 100, 2).to(device=device)

    x_train, y_train, x_val, y_val = data
    # print(x_train.shape)
    # print(y_train.shape)

    x_train = torch.from_numpy(x_train).to(device)
    y_train = torch.from_numpy(y_train).to(device)
    x_val = torch.from_numpy(x_val).to(device)
    y_val = torch.from_numpy(y_val).to(device)



    train_losses = []
    val_losses = []


    print('---------- Training has started: -------------')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # DEFINE YOUR OPTIMIZER



    for epoch in iterator:  # DO NOT CHANGE THIS LINE


        y_pred = net(x_train)
       ## print(y_pred.shape)
        # y_train = y_train.type(torch.LongTensor)

       ## print(y_train)

        loss = my_NLLloss(y_pred, y_train)
        #maybe you need to do net.zero_grad()

        ##loss.requires_grad_()

       ## print("THIS IS LOSS:")
       ## print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #
        train_loss = loss

        # for x_, y_ in zip(x_val, y_val):
        #     x = torch.from_numpy(x_).to(device=device)
        #     y = torch.from_numpy(np.asarray(y_)).to(device=device)
        #     print(x.shape)
        #     scores = net(x)
        #     loss = loss_func(scores, y)
        #     optimizer.zero_grad()
        #
        #     loss.backward()
        #     optimizer.step()

        #
        y_pred = net(x_val)
        loss = my_NLLloss(y_pred, y_val)

       ## loss.requires_grad_()
        # optimizer.zero_grad()
        #
        # loss.backward()
        # optimizer.step()
        val_loss = loss


        # <<< END YOUR CODE

        # DO NOT MODIFY THE BELOW
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())
        if tqdm is not None:
            iterator.set_description(f' Epoch: {epoch + 1}')
            iterator.set_postfix(train_loss=round(train_loss.item(), 1),
                                 val_loss=round(val_loss.item(), 1))
        else:
            print(
                f'epoch {epoch + 1}: train_loss = {train_loss}, val_loss = {val_loss}')

    print('---------- Training ended. -------------\n')

    plt.figure()
    epochs = list(range(max_epochs))
    plt.plot(epochs[5:], train_losses[5:])
    plt.plot(epochs[5:], val_losses[5:])
    plt.legend(['train', 'val'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and Validation Losses')
    plt.savefig(os.path.join(os.path.dirname(__file__), 'training_curve.png'))
    plt.close()

    return net, train_losses[-1], val_losses[-1]


def plot_predictions(model: nn.Module, stock_dict: dict) -> None:
    """
    Given a trained model and a dictionary of stock dataframes, predict the
    stock 'Close' prices for each stock. Plot the predicted 'Close' prices vs.
    the actual 'Close' prices for each stock (only plot the first 50 data samples).

    Check the handout for an example.

    Args:
        model (nn.Module): a trained model
        stock_dict (dict): a dictionary of stock dataframes

    Returns:
        None
    """

    fig, axs = plt.subplots(2, 2)  # axs may be useful
    fig.tight_layout(pad=3.0)  # give some space between subplots

    for k, stock in enumerate(list(stock_dict.keys())):
        (_, _, x_val, y_val) = split_stock(stock_dict[stock])

        pred = model(torch.Tensor(x_val).to(device)).detach().cpu().numpy()

        pred_prices, pred_risks = pred[:, 0], np.sqrt(np.exp(pred[:, 1]))
        rmse = np.sqrt(np.mean((pred_prices - y_val) ** 2))
        print(f'RMSE for {stock} is: {rmse}')

        i, j = k // 2, k % 2

        prices_range = [pred_prices - pred_risks, pred_prices + pred_risks]
        axs[i, j].plot(y_val[:50])
        axs[i, j].plot(pred_prices[:50])
        axs[i, j].legend(['real', 'pred'])
        axs[i, j].fill_between(list(range(50)), prices_range[0]
        [:50], prices_range[1][:50], color=None, alpha=.15)
        axs[i, j].set_title(f'{stock}')

    plt.savefig(os.path.join(os.path.dirname(__file__), 'predictions.png'))
    print('Predictions plotted.')




if __name__ == '__main__':
    os.system('cls' if os.name == 'nt' else 'clear')

    file_path = os.path.join(os.path.dirname(
        __file__), 'datasets/stock_train.csv')
    stock_dict = pre_process(file_path)

    plot_data(stock_dict)

    data = get_train_valid(stock_dict)

    net, train_loss, val_loss = train(data, max_epochs=1000)
    plot_predictions(net, stock_dict)
