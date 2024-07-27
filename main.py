import asyncio
import numpy as np
from abc import ABC, abstractmethod
from gym.spaces import Box, Space
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.data import GenData
from poke_env.environment.status import Status
from poke_env.player import (
	Player,
	ObservationType,
	MaxBasePowerPlayer,
)
from poke_env import PlayerConfiguration
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from itertools import count


class LivePlayer(Player):
	def __init__(self, battle_format, policy, weighted_random, player_name, **kwargs):
		rand_id = str(random.randint(0, 2048))
		player_name += rand_id
		print(f"Initializing {player_name}")
		super(LivePlayer, self).__init__(player_configuration=PlayerConfiguration(f"{player_name}", ""), battle_format=battle_format, **kwargs)
		self.policy = policy
		self.weighted_random = weighted_random
		self.m = nn.Sigmoid()

	def action_to_move(self, action: int, battle: AbstractBattle):
		if action == -1:
			print("we are forfeiting")
			return ForfeitBattleOrder()
		elif (
			action < 4
			and action < len(battle.available_moves)
			and not battle.force_switch
		):
			return self.create_order(battle.available_moves[action])
		elif (
			not battle.force_switch
			and battle.can_z_move
			and battle.active_pokemon
			and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
		):
			return self.create_order(
				battle.active_pokemon.available_z_moves[action - 4], z_move=True
			)
		elif (
			battle.can_mega_evolve
			and 0 <= action - 8 < len(battle.available_moves)
			and not battle.force_switch
		):
			return self.create_order(
				battle.available_moves[action - 8], mega=True
			)
		elif (
			battle.can_dynamax
			and 0 <= action - 12 < len(battle.available_moves)
			and not battle.force_switch
		):
			return self.create_order(
				battle.available_moves[action - 12], dynamax=True
			)
		elif 0 <= action - 16 < len(battle.available_switches):
			return self.create_order(battle.available_switches[action - 16])
		else:
			return self.choose_random_move(battle)

	def choose_move(self, battle):
		state = torch.tensor(self.embed_battle(battle))
		with torch.no_grad():
			move_weights = self.policy(state)
			n_actions = move_weights.shape[0]
			if self.weighted_random: # testing, not sure if this works
				move_weights = self.m(move_weights) # normalize
				move_rankings = torch.multinomial(move_weights, n_actions)
			else:
				move_rankings = move_weights.topk(k=n_actions, dim=0).indices
			for move in move_rankings:
				if self.check_valid(move.item(), battle):
					return self.action_to_move(move.view(1, 1), battle)

	def check_valid(self, action: int, battle: AbstractBattle):
		if action == -1:
			return False
		elif (
			action < 4
			and action < len(battle.available_moves)
			and not battle.force_switch
		):
			return True
		elif (
			not battle.force_switch
			and battle.can_z_move
			and battle.active_pokemon
			and 0 <= action - 4 < len(battle.active_pokemon.available_z_moves)
		):
			return True
		elif (
			battle.can_mega_evolve
			and 0 <= action - 8 < len(battle.available_moves)
			and not battle.force_switch
		):
			return True
		elif (
			battle.can_dynamax
			and 0 <= action - 12 < len(battle.available_moves)
			and not battle.force_switch
		):
			return True
		elif 0 <= action - 16 < len(battle.available_switches):
			return True
		else:
			return False

class MyRLPlayer(LivePlayer):
	def embed_battle(self, battle: AbstractBattle) -> ObservationType:
		moves_base_power = -np.ones(4)
		moves_dmg_multiplier = np.ones(4)
		moves_have_status = np.zeros(4)
		for i, move in enumerate(battle.available_moves):
			moves_base_power[i] = (
				move.base_power / 100
			)
			if move.type:
				moves_dmg_multiplier[i] = move.type.damage_multiplier(
					battle.opponent_active_pokemon.type_1,
					battle.opponent_active_pokemon.type_2,
					type_chart=type_chart
				)
			if move.status is not None:
				moves_have_status[i] = 1
		opponent_status = np.zeros(len(Status))
		if battle.opponent_active_pokemon.status is not None:
			opponent_status[battle.opponent_active_pokemon.status.value - 1] = 1

		final_vector = np.concatenate(
			[
				moves_base_power,
				moves_dmg_multiplier,
				opponent_status,
				moves_have_status,
			]
		)
		return np.float32(final_vector)

	def describe_embedding(self) -> Space:
		low =\
			  [-1, -1, -1, -1, 0, 0, 0, 0] +\
			  [0 for _ in range(len(Status))] +\
			  [0 for _ in range(4)]
		high =\
			   [3, 	3,  3,  3, 4, 4, 4, 4] +\
			   [0 for _ in range(len(Status))] +\
			   [1 for _ in range(4)]
		return Box(
			np.array(low, dtype=np.float32),
			np.array(high, dtype=np.float32),
			dtype=np.float32,
		)

class DQN(nn.Module):
	def __init__(self, n_observations, n_actions, n_hidden):
		super(DQN, self).__init__()
		self.layer1 = nn.Linear(n_observations, 128)
		self.layer2 = nn.Linear(128, n_hidden)
		self.layer3 = nn.Linear(n_hidden, 128)
		self.layer4 = nn.Linear(128, n_actions)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		return self.layer4(x)


data = GenData(8)
type_chart = data.type_chart

async def main():

	n_challenges = 3
	n_completed = 0
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	policy_net = DQN(19, 22, 389).to(device)
	policy_net.load_state_dict(torch.load("models/y7v7nc.pth"))

	my_player = MyRLPlayer("gen8randombattle", policy_net, weighted_random=False, player_name="MyRLPlayer")
	await my_player.accept_challenges(opponent=None, n_challenges=n_challenges)


if __name__ == '__main__':
	asyncio.get_event_loop().run_until_complete(main())

