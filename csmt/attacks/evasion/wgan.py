'''
Author: your name
Date: 2021-03-25 15:30:53
LastEditTime: 2021-07-10 19:13:20
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /NashHE/nashhe/ml_ids/train_wgan.py
'''
import sys
from os import path
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import itertools
import os
import numpy as np
from csmt.attacks.evasion.abstract_evasion import AbstractEvasion

class Generator(nn.Module):
  """Generator in Wasserstein GAN."""
  def __init__(self, input_size, output_size):
    """Create a generator."""
    super(Generator, self).__init__()

    def block(input_dim, output_dim):
      layers = [nn.Linear(input_dim, output_dim)]
      layers.append(nn.ReLU(inplace=False))
      return layers

    self.model = nn.Sequential(
      *block(input_size, 128),
      *block(128, 128),
      *block(128, 128),
      nn.Linear(128, output_size),
    )
  def forward(self, x):
    """Do a forward pass."""
    adversarial_traffic = self.model(x)
    return adversarial_traffic

class Discriminator(nn.Module):
  """Discriminator in Wasserstein GAN."""
  def __init__(self, input_size):
    """Create a discriminator."""
    super(Discriminator, self).__init__()

    def block(input_dim, output_dim):
      layers = [nn.Linear(input_dim, output_dim)]
      layers.append(nn.LeakyReLU(inplace=False))
      return layers

    self.model = nn.Sequential(
      *block(input_size, 128),
      *block(128, 128),
      *block(128, 128),
      nn.Linear(128, 1)
    )

  def forward(self, x):
    """Do a forward pass."""
    traffic = self.model(x)
    return traffic

class WGAN():
    def __init__(self,n_features,noise_dim=9,epochs=100,batch_size=64,learning_rate=0.0001,weight_clipping=0.01,critic_iter=5,evaluate=200):
        self.n_features=n_features
        self.noise_dim=noise_dim
        self.epochs=epochs
        self.batch_size=batch_size
        self.learning_rate=learning_rate
        self.weight_clipping=weight_clipping
        self.critic_iter=critic_iter
        self.evaluate=evaluate
        self.generator = Generator(self.n_features + self.noise_dim, self.n_features)
        self.discriminator = Discriminator(self.n_features)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        self.optim_G = optim.RMSprop(self.generator.parameters(), lr=self.learning_rate)
        self.optim_D = optim.RMSprop(self.discriminator.parameters(), lr=self.learning_rate)

        self.start_epoch = 0

    def train(self,trainingset):
        self.generator.train()
        self.discriminator.train()
        normal_traffic, normal_labels, malicious_traffic, malicious_labels = self._extract_dataset(trainingset)

        epoch_iterator = self._get_epoch_iterator()
        for epoch in epoch_iterator:
            self._require_grad(self.discriminator, True)
            self._require_grad(self.generator, False)
            # Discriminator training
            for c in range(self.critic_iter):
                normal_traffic_batch = self._sample_normal_traffic(normal_traffic)
                malicious_traffic_batch = self._sample_malicious_traffic(malicious_traffic)
                adversarial_traffic = self.generator(malicious_traffic_batch)
                discriminated_normal = torch.mean(self.discriminator(normal_traffic_batch)).view(1)
                discriminated_adversarial = torch.mean(self.discriminator(adversarial_traffic)).view(1)
                discriminator_loss = - (discriminated_normal - discriminated_adversarial)
                self.optim_D.zero_grad()
                discriminator_loss.backward()
                self.optim_D.step()
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.weight_clipping, self.weight_clipping)
            # Generator training
            self._require_grad(self.discriminator, False)
            self._require_grad(self.generator, True)
            malicious_traffic_batch = self._sample_malicious_traffic(malicious_traffic)
            adversarial_traffic = self.generator(malicious_traffic_batch) # 64*23
            generator_objective = torch.mean(self.discriminator(adversarial_traffic)).view(1)
            generator_loss = - generator_objective
            self.optim_G.zero_grad()
            generator_loss.backward()
            self.optim_G.step()

            if epoch % self.evaluate == 0:
                self.generator.eval()
                self.discriminator.eval()
                self.generator.train()
                self.discriminator.train()
            

    def save(self, path):
        """Save model."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.generator.state_dict(), path + '/generator.pt')
        torch.save(self.discriminator.state_dict(), path + '/discriminator.pt')

    def load(self, path):
        """Load model from a file."""
        self.generator.load_state_dict(torch.load(path + '/generator.pt'))
        self.discriminator.load_state_dict(torch.load(path + '/discriminator.pt'))
          
    def _extract_dataset(self, dataset):
        normal_traffic, malicious_traffic, normal_labels, malicious_labels = dataset
        normal_traffic, malicious_traffic,normal_labels, malicious_labels=normal_traffic.values, malicious_traffic.values,normal_labels.values, malicious_labels.values
        normal_traffic_tensor = torch.tensor(normal_traffic, dtype=torch.float, requires_grad=True).to(self.device)
        malicious_traffic_tensor = torch.tensor(malicious_traffic, dtype=torch.float, requires_grad=True).to(self.device)
        return normal_traffic_tensor, normal_labels, malicious_traffic_tensor, malicious_labels

    def _get_epoch_iterator(self):
      if self.epochs < 0:
          return itertools.count(self.start_epoch)
      else:
          return range(self.start_epoch, self.epochs)

    def _require_grad(self, module, require):
        for parameter in module.parameters():
            parameter.requires_grad = require

    def _sample_normal_traffic(self, traffic):
        indices = np.random.randint(0, len(traffic), self.batch_size)
        return traffic[indices]

    def _sample_malicious_traffic(self, traffic):
        indices = np.random.randint(0, len(traffic), self.batch_size)
        batch = traffic[indices]
        noise = torch.rand(self.batch_size, self.noise_dim).to(self.device)
        batch_with_noise = torch.cat((batch, noise), 1)
        return batch_with_noise

    def predict(self, traffic):
        """Use discriminator to predict whether real or fake."""
        outputs = self.discriminator(traffic).squeeze()
        predictions = torch.empty((len(outputs),), dtype=torch.uint8)
        predictions[outputs < 0] = 0   # adversarial traffic
        predictions[outputs >= 0] = 1  # normal traffic
        return predictions.cpu().numpy()

    def generate(self, malicious_traffic):
        """Generate adversarial traffic."""
        self.generator.eval()
        self.discriminator.eval()
        n_observations_mal = len(malicious_traffic)
        batch_Malicious = torch.Tensor(malicious_traffic)
        noise = torch.rand(n_observations_mal, self.noise_dim)
        batch_Malicious_noise = torch.cat((batch_Malicious, noise), 1)
        batch_Malicious_noise = Variable(batch_Malicious_noise.to(self.device))
        adversarial = self.generator(batch_Malicious_noise)
        return adversarial

class WGANEvasionAttack(AbstractEvasion):
    def __init__(self,datasets_name):
        self.datasets_name=datasets_name
    def generate(self,X):
        model_path=path.join('csmt/attacks/evasion/saved_models/',self.datasets_name)
        n_features,trainingset,testingset=get_mimicry_datasets(self.datasets_name)
        wgan=WGAN(n_features)

        wgan.train(trainingset)
        wgan.save(model_path)

        wgan.load(model_path)
        X_adv=wgan.generate(X).detach()
        return X_adv
