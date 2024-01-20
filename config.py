from dataclasses import dataclass

@dataclass
class Params:
    num_agents: int
    grid_size: int 
    num_steps: int