# Escape Solaire, but don't forget your Estus. Reinforcement Learning

Este projeto implementa um algoritmo de Aprendizado por Reforço para um cenário inspirado no jogo "Dark Souls" para o trabalho final de disciplina de Inteligência Artifical do curso de Ciências da Computação da Universidade Regional Integrada do Alto Uruguai e das Missões. O objetivo do agente é sobreviver em um ambiente ameaçador, coletando suprimentos (Estus Flask) e evitando os Hollows (inimigos).

O projeto utiliza a estrutura de dados *Q-table* do método *Q-learning* para o Aprendizado por Reforço do agente, junto de bibliotecas como *numpy*, para processar as matrizes, e *pygame*, para a visualização da execução e treinamento do agente.

## Instalação

### Pré-requisitos

- Python 3.12.4
- pip (gerenciador de pacotes do Python)

### Passos para Instalação

1. Clone o repositório:
   ```bash   
   git clone https://github.com/Lowez/IA-TrabalhoFinal.git
   cd IA-TrabalhoFinal

2. Criar um ambiente virtual para rodar o arquivo
   python3 -m venv venv        # Linux e macOS
   python -m venv env        # Windows

3. Ativar o ambiente
   source env/bin/activate     # Linux e macOS
   .\env\Scripts\activate    # Windows

4. Instalar bibliotecas necessárias
   pip install -r requirements.txt

5. Executar o código
   python index.py

Obs.: Para treinar invés de apenas executar, remover o arquivo `q_table_and_environment.json`