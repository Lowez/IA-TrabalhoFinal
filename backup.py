import numpy as np
import random
import pygame
import sys
import json

# Classe e definições do ambiente
class Environment:
    def __init__(self, size=10, num_boxes=8, num_supplies=5, num_enemies=3):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int)
        self.start_state = (0, 0)
        self.goal_state = (size-1, size-1)
        self.hole_states = self._place_random_items(num_boxes)
        self.supply_states = self._place_random_items(num_supplies, exclude=self.hole_states)
        self.obstacle_states = self._place_random_items(num_enemies, exclude=self.hole_states | self.supply_states)
        self.supplies_collected = set()
        self._update_grid()
        self._load_images()

    def _place_random_items(self, num_items, exclude=set()):
        items = set()
        while len(items) < num_items:
            i, j = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if (i, j) not in items and (i, j) not in exclude and (i, j) != self.start_state and (i, j) != self.goal_state:
                items.add((i, j))
        return items

    def _update_grid(self):
        self.grid.fill(0)
        for i, j in self.hole_states:
            self.grid[i][j] = 1
        for i, j in self.supply_states:
            self.grid[i][j] = 2
        for i, j in self.obstacle_states:
            self.grid[i][j] = 3

    def _load_images(self):
        self.agent_img = pygame.image.load('./assets/survival.png')
        self.goal_img = pygame.image.load('./assets/porta.png')
        self.hole_img = pygame.image.load('./assets/box.jpg')
        self.supply_img = pygame.image.load('./assets/estus.png')
        self.obstacle_img = pygame.image.load('./assets/hollow2.png')
        self.background_img = pygame.image.load('./assets/grass.jpg')

    def reset(self):
        self.current_state = self.start_state
        self.supplies_collected = set()
        return self.current_state, tuple(self.supplies_collected)

    def step(self, action):
        i, j = self.current_state
        next_state = self._move(i, j, action)
        self.current_state = self._validate_move(next_state)
        
        reward, done = self._compute_reward()
        return self.current_state, tuple(self.supplies_collected), reward, done

    def _move(self, i, j, action):
        if action == 0: # move up
            i = max(i-1, 0)
        elif action == 1: # move down
            i = min(i+1, self.size-1)
        elif action == 2: # move left
            j = max(j-1, 0)
        elif action == 3: # move right
            j = min(j+1, self.size-1)
        return (i, j)

    def _validate_move(self, next_state):
        if next_state in self.obstacle_states:
            return self.current_state
        return next_state

    def _compute_reward(self):
        if self.current_state == self.goal_state:
            if len(self.supplies_collected) == len(self.supply_states):
                return 10, True
            else:
                return -1, False
        elif self.current_state in self.hole_states:
            return -5, True
        elif self.current_state in self.supply_states and self.current_state not in self.supplies_collected:
            self.supplies_collected.add(self.current_state)
            return 2, False
        else:
            return -0.1, False

    def render(self, screen, cell_size=60):
        screen.fill((200, 200, 200))
        for i in range(self.size):
            for j in range(self.size):
                self._draw_item(screen, i, j, cell_size)
        pygame.display.flip()

    def _draw_item(self, screen, i, j, cell_size):
        screen.blit(pygame.transform.scale(self.background_img, (cell_size, cell_size)), (j * cell_size, i * cell_size))
        if (i, j) == self.current_state:
            img = self.agent_img
        elif (i, j) == self.goal_state:
            img = self.goal_img
        elif self.grid[i][j] == 1:
            img = self.hole_img
        elif self.grid[i][j] == 2 and (i, j) not in self.supplies_collected:
            img = self.supply_img
        elif self.grid[i][j] == 3:
            img = self.obstacle_img
        else:
            img = None
        
        if img:
            screen.blit(pygame.transform.scale(img, (cell_size, cell_size)), (j * cell_size, i * cell_size))

    def save_q_table(self, q_table, filename='q_table_and_environment.json'):
        data = {
            'q_table': q_table.tolist(),
            'environment_info': {
                'size': self.size,
                'num_boxes': list(self.hole_states),
                'num_supplies': list(self.supply_states),
                'num_enemies': list(self.obstacle_states),
                'start_state': self.start_state,
                'goal_state': self.goal_state
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f)
        print(f"Tabela Q e informações do ambiente salvas em {filename}")

    def load_q_table(self, filename='q_table_and_environment.json'):
        with open(filename, 'r') as f:
            data = json.load(f)
            q_table = np.array(data['q_table'])
            environment_info = data['environment_info']
            
            # Reconstruindo o ambiente com base nas informações salvas
            self.size = environment_info['size']
            self.start_state = tuple(environment_info['start_state'])
            self.goal_state = tuple(environment_info['goal_state'])
            self.hole_states = set(tuple(pos) for pos in environment_info['num_boxes'])
            self.supply_states = set(tuple(pos) for pos in environment_info['num_supplies'])
            self.obstacle_states = set(tuple(pos) for pos in environment_info['num_enemies'])
            self.supplies_collected = set()
            self._update_grid()
            self._load_images()
            
            print(f"Tabela Q carregada de {filename}")
            
            return q_table

def initialize_q_table(environment):
    return np.zeros((environment.size, environment.size, 2 ** len(environment.supply_states), 4))

def epsilon_greedy_policy(state, collected_supplies, q_table, environment, epsilon):
    supply_index = get_supply_index(collected_supplies, environment)
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(q_table[state[0]][state[1]][supply_index])

def get_supply_index(collected_supplies, environment):
    return int(''.join(['1' if (i, j) in collected_supplies else '0' for (i, j) in environment.supply_states]), 2)

def train_agent(environment, q_table, screen, cell_size, num_episodes=10000, max_steps_per_episode=100, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, min_epsilon=0.01, epsilon_decay_rate=0.001):
    for episode in range(num_episodes):
        state, collected_supplies = environment.reset()
        done = False
        t = 0
        while not done and t < max_steps_per_episode:
            action = epsilon_greedy_policy(state, collected_supplies, q_table, environment, epsilon)
            next_state, next_collected_supplies, reward, done = environment.step(action)
            update_q_table(q_table, state, collected_supplies, action, reward, next_state, next_collected_supplies, environment, learning_rate, discount_factor)
            state, collected_supplies = next_state, next_collected_supplies
            t += 1
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay_rate))
        if episode % 1000 == 0:
            print(f'Episode: {episode}')
            environment.render(screen, cell_size)
    
    # Após o treinamento, salvar a tabela Q e informações do ambiente
    # TODO DESCOMENTAR LINHA ABAIXO PARA SALVAR ARQUIVO DE TREINAMENTO
    environment.save_q_table(q_table)

def update_q_table(q_table, state, collected_supplies, action, reward, next_state, next_collected_supplies, environment, learning_rate, discount_factor):
    supply_index = get_supply_index(collected_supplies, environment)
    next_supply_index = get_supply_index(next_collected_supplies, environment)
    q_table[state[0]][state[1]][supply_index][action] += learning_rate * \
        (reward + discount_factor * np.max(q_table[next_state[0]][next_state[1]][next_supply_index]) - q_table[state[0]][state[1]][supply_index][action])

def test_agent(environment, q_table, screen, cell_size=60):
    state, collected_supplies = environment.reset()
    done = False
    while not done:
        supply_index = get_supply_index(collected_supplies, environment)
        action = np.argmax(q_table[state[0]][state[1]][supply_index])
        next_state, next_collected_supplies, reward, done = environment.step(action)
        environment.render(screen, cell_size)
        pygame.time.wait(500)
        state, collected_supplies = next_state, next_collected_supplies

if __name__ == "__main__":
    # Criação da instância do ambiente
    environment = Environment(size=7, num_boxes=4, num_supplies=3, num_enemies=6)
    
    # Pygame setup
    pygame.init()
    cell_size = 60
    screen = pygame.display.set_mode((environment.size * cell_size, environment.size * cell_size))
    pygame.display.set_caption('IA Trabalho Final')
    
    # Parâmetros de treinamento
    num_episodes = 20000
    max_steps_per_episode = 200
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 1.0
    min_epsilon = 0.01
    epsilon_decay_rate = 0.001

    # Tenta encontrar arquivo de último treinamento, se encontrar, executa o agente, senão, treina-o novamente
    try:
        q_table = environment.load_q_table()
        test_agent(environment, q_table, screen, cell_size)
    except FileNotFoundError:
        q_table = initialize_q_table(environment)
        print("Inicializando nova tabela Q.")
        train_agent(environment, q_table, screen, cell_size, num_episodes, max_steps_per_episode, learning_rate, discount_factor, epsilon, min_epsilon, epsilon_decay_rate)
        test_agent(environment, q_table, screen, cell_size)

    pygame.quit()
    sys.exit()
