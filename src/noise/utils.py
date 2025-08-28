
import torch
from torch import IntTensor, Tensor
import torch.nn as nn

class CoinChange:

    def __init__(self, coins, max_amount=10):
        self.coins = sorted(coins, reverse=True)
        self.coins_dict = {coin: i for i, coin in enumerate(self.coins)}
        self.max_amount = 0
        self.filled_amount = 0

        self.coins_used = [0]
        self.next_coin = [-1]

        self.increase_max_amount(max_amount)
        self.compute()
        not_valid = self.validate_coins()
        if not_valid:
            print('Cannot obtain some values, invalid coins: {}'.format(coins))


    def validate_coins(self):
        return any([num_coins < 0 for num_coins in self.coins_used])


    def increase_max_amount(self, max_amount):

        if self.max_amount < max_amount:
            
            self.coins_used.extend([-1] * (max_amount - self.max_amount))
            self.next_coin.extend([-1] * (max_amount - self.max_amount))

            self.max_amount = max_amount


    def compute(self):

        for amount in range(self.filled_amount + 1, self.max_amount + 1):

            min_coins_num = float('inf')
            used_coin = -1
            for coin in self.coins:
                next_amount = amount - coin
                if next_amount < 0:
                    continue
                elif self.coins_used[next_amount] >= 0 and \
                        self.coins_used[next_amount] < min_coins_num:
                    min_coins_num = self.coins_used[next_amount]
                    used_coin = coin
                
            if used_coin > 0:
                self.coins_used[amount] = min_coins_num + 1
                self.next_coin[amount] = used_coin
            

        self.filled_amount = self.max_amount


    def expand(self, max_amount):
        if max_amount > self.max_amount:
            self.increase_max_amount(max_amount)
            self.compute()


    def get_coins(self, amount):
        """Return the coins used to make up the amount
        """
        if amount > self.max_amount:
            self.increase_max_amount(amount)
            self.compute()
        
        coins = []
        while amount > 0:
            coins.append(self.next_coin[amount])
            amount -= self.next_coin[amount]

        return coins
    
    def get_next_coin(self, amount):
        """Return the next coin used to make up the amount
        """
        if amount > self.max_amount:
            self.increase_max_amount(amount)
            self.compute()
        
        return self.next_coin[amount]
    
    def get_next_coin_idx(self, amount, as_hist=False):
        """Return the index of the next coin used to make up the amount
        """
        next_coin = self.get_next_coin(amount)
        next_coin_idx = self.coins_dict[next_coin]
        
        if as_hist:
            histogram = [0] * len(self.coins)
            histogram[next_coin_idx] = 1
            return histogram
        else:
            return next_coin_idx
    
    
    def get_coins_used(self, amount):
        """Return the number of coins used to make up the amount
        """
        if amount > self.max_amount:
            self.increase_max_amount(amount)
            self.compute()
        
        return self.coins_used[amount]
    
    def get_coins_map(self):
        return self.coins
    
    
    def get_coins_histogram(self, amount):
        """Return the histogram of coins used to make up the amount
        """
        if amount > self.max_amount:
            self.increase_max_amount(amount)
            self.compute()
        
        histogram = [0] * len(self.coins)
        while amount > 0:
            next_coin = self.next_coin[amount]
            if next_coin == -1:
                return histogram
            histogram[self.coins_dict[next_coin]] += 1
            amount -= self.next_coin[amount]

        return histogram

    
import numpy as np
from scipy.stats import multivariate_hypergeom


def compute_next_histograms(hists, small_to_big=True):

    device = hists.device

    # get mask where histograms are zero (no coins used)
    zero_mask = hists == 0

    # histograms are already ordered from biggest to smallest coin
    numel = hists.numel()
    if small_to_big:
        range_interv = (1, numel+1, 1) # max is the smallest
    else:
        range_interv = (numel+1, 1, -1) # max is the biggest

    range_tensor = torch.arange(*range_interv, dtype=torch.int32, device=device)
    range_tensor = range_tensor.view(hists.shape)
    range_tensor[zero_mask] = 0 # zero will always be the min, never picked
    next_idx = torch.argmax(range_tensor, dim=-1, keepdim=True)
    hists_norm = torch.zeros_like(hists)
    #hists_norm[next_idx] = 1
    hists_norm.scatter_(-1, next_idx, 1)

    return hists_norm


def compute_choose_t_coins_from_hists(hists, t, small_to_big=True):

    cum_hists = torch.cumsum(hists, dim=-1)

    if small_to_big: # reverse the cumulation
        cum_hists = hists + cum_hists[..., -1:] - cum_hists

    if isinstance(t, Tensor):
        t = t.unsqueeze(-1)

    suppressed_cum_hists = torch.clip(cum_hists - t, min=0)
    delta = hists - suppressed_cum_hists

    return torch.clip(delta, min=0)


class DistCoinChange(nn.Module):

    def __init__(self, coins, max_amount=100, dist_type='histogram'):
        """dist type can be:
        - histogram: the distribution of coins used to make up the amount
        - next: the next coin in the sequence used to make up the amount, order is from small to big
        - next_rev: the next coin in the sequence used to make up the amount, order is from big to small
        """
        super().__init__()

        self.coin_change = CoinChange(coins, max_amount)
        self.dist_type = dist_type

        if 'next' in self.dist_type:
            hist_shape = (1, len(coins), 2)
        else:
            hist_shape = (1, len(coins))

        self.register_buffer('histograms', torch.zeros(hist_shape, dtype=torch.float32), persistent=False)
        self.register_buffer('int_histograms', torch.zeros((1, len(coins)), dtype=torch.int32), persistent=False)
        self.register_buffer('coins_used', torch.zeros((1,), dtype=torch.int32), persistent=False)
        self.register_buffer('coins_map', torch.tensor(self.coin_change.get_coins_map(), dtype=torch.int32), persistent=False)

        self.update_histograms(max_amount)


    def get_onehot_coin(self, coin):
        return self.int_histograms[coin].float()


    def update_histograms(self, new_max_amount):

        curr_max_amount = self.histograms.shape[0] - 1

        # don't need to update if no new amount is added
        if curr_max_amount >= new_max_amount:
            return

        # expand the coin change
        self.coin_change.expand(new_max_amount)

        # update the histograms
        device = self.histograms.device

        # collect integer histograms (count of coins used per denomination)
        new_hists = []
        coins_used = []
        for amount in range(curr_max_amount + 1, self.coin_change.max_amount + 1):
            coins_used.append(self.coin_change.get_coins_used(amount))
            new_hists.append(self.coin_change.get_coins_histogram(amount))

        # convert to tensors
        new_hists = torch.tensor(new_hists, dtype=torch.float32, device=device)
        coins_used = torch.tensor(coins_used, dtype=torch.int32, device=device)

        # normalize histograms (or compute next coin)
        if 'next' in self.dist_type:
            new_hists_norm = compute_next_histograms(new_hists, small_to_big=self.dist_type=='next')
            new_hists_norm_rev = compute_next_histograms(new_hists, small_to_big=self.dist_type=='next_rev')
            new_hists_norm = torch.stack((new_hists_norm, new_hists_norm_rev), dim=-1)

        elif self.dist_type == 'histogram':
            new_hists_norm = new_hists / coins_used.unsqueeze(-1)

        # update buffers
        self.coins_used = torch.cat((self.coins_used, coins_used), dim=0)
        self.histograms = torch.cat((self.histograms, new_hists_norm), dim=0)
        if 'next' in self.dist_type:
            self.rev_histograms = torch.cat((self.int_histograms, new_hists_norm_rev.int()), dim=0)
        self.int_histograms = torch.cat((self.int_histograms, new_hists.int()), dim=0)


    def safety_check(self, amounts: IntTensor):
        if isinstance(amounts, int):
            new_amount = amounts
        else:
            new_amount = amounts.max().item()
        self.update_histograms(new_amount * 2)

    
    def forward(self, amounts: IntTensor, safe=True, reverse=False):
        """Return the distribution of coins used to make up the amounts
        Warning: this can be unsafe if a safety check has not been performed
        """
        if safe:
            self.safety_check(amounts)
        if 'next' in self.dist_type:
            if reverse:
                out = self.histograms[amounts, :, 1]
            else:
                out = self.histograms[amounts, :, 0]
        else:
            out = self.histograms[amounts]
        return out
    
    def forward_int(self, amounts: IntTensor, safe=True):
        """Return the distribution of coins used to make up the amounts
        Warning: this can be unsafe if a safety check has not been performed
        """
        if safe:
            self.safety_check(amounts)
        out = self.int_histograms[amounts]
        return out
    

    def get_coins_used(self, amounts: IntTensor, safe=True):
        """Return the number of coins used to make up the amounts
        Warning: this can be unsafe if a safety check has not been performed
        """
        if safe:
            self.safety_check(amounts)
        out = self.coins_used[amounts]
        return out
    

    def sample_categorical(self, probs=None, logits=None):
        """Return a sample of coins used to make up the amounts
        Warning: this can be unsafe if a safety check has not been performed
        """

        sampled_categories = torch.distributions.categorical.Categorical(
            probs = probs,
            logits = logits
        ).sample()
        
        return self.coins_map[sampled_categories]
    

    def sample_categorical_from_amounts(self, amounts: IntTensor, safe=True, reverse=False):
        mask = amounts == 0
        probs = self(amounts, safe=safe, reverse=reverse)
        probs[mask, 0] = 1.
        sampled_coins = self.sample_categorical(probs=probs)
        sampled_coins[mask] = 0
        return sampled_coins
    

    def sample_multivariate_hypergeometric(self, amounts: IntTensor, t: IntTensor, safe=True):
        """Return a sample of coins used to make up the amounts
        Warning: this can be unsafe if a safety check has not been performed
        """
        if self.dist_type == 'histogram':
            # have to use scipy, as implementing this stuff in pytorch
            # is too difficult, and maybe not worth it
            hists = self.forward_int(amounts, safe=safe).cpu().numpy()
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()

            # sample
            sampled_categories = multivariate_hypergeom.rvs(
                m = hists,  # number of balls in the urn for each color
                n = t       # extract t balls from the urn
            )
            # sampled categories is the number of balls of each color extracted
            sampled_categories = torch.from_numpy(sampled_categories).to(device=self.histograms.device, dtype=torch.float32)

        if self.dist_type == 'next':

            # will choose an amount t of coins from the histograms
            # e.g: amount is 9, with denominations [1, 4], at t=2
            # its histogram is: [2, 1] if the order of denominations is [4, 1]
            # then:
            # - if we choose 2 coins from the histogram from small to big
            #   the result will be: [1, 1] (1 of value 4, 1 of value 1)
            # - if we choose 2 coins from the histogram from big to small
            #   the result will be: [2, 0] (2 of value 4, 0 of value 1)
            sampled_categories = compute_choose_t_coins_from_hists(
                hists = self.forward_int(amounts, safe=safe),
                t = t,
                small_to_big = self.dist_type == 'next'
            ).float()


        # weight each coin by the number of times it was sampled by the hypergeometric
        # and also sum everything up
        sampled_num = torch.inner(sampled_categories, self.coins_map.float()).int()


        return sampled_num
    


def main():

    d = DistCoinChange([1, 2, 4])

    print(d.get_onehot_coin(torch.tensor([4, 2, 1, 4])))


if __name__ == '__main__':
    main()