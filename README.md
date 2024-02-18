# SimulationSandbox

Minimalistic simulation environment implemented in Jax with simple server-client interaction. Enables modifying the simulation state in real time from notebook controllers, and plotting the state on distant clients. 

## Install 

Get the repo :

```bash
git clone git@github.com:corentinlger/SimulationSandbox.git
cd SimulationSandox/
```

Setup a virtual environment and install the dependencies :

```bash
python3 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

## Usage 

You can run a simulation on your machine (using either the [2D](simulationsandbox/environments/two_d_example_env.py) or [3D](simulationsandbox/environments/three_d_example_env.py) example envs provided):

```bash
python3 simulate.py  
```

You can also easiely implement your own environment and add it to the [environments directory](simulationsandbox/environments/). 

Or host it on a server :

```bash
python3 server.py
```

Visualize it on distant clients : 

```bash
python3 client.py
```

Modify the state of the simulation in real time : use [this notebook client](notebook_controller.ipynb)


## Tests

You can test your code locally by running : 

```bash
pytest
```

And add your own features to the [tests directory](tests/)


## TODO : 

- Add a networking class for client-server interaction
- Stop the atrocious popping up of matplotlib interactive figures 
- Replace sending the first example state with pickle by generating it on the client side 
- Further check the interaction between 3D sim and notebook client 