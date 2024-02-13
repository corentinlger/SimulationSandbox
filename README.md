# SimulationSandbox

Minimalistic simulation environment with simple server-client interaction. Enables modifying the simulation state in real time from notebook controllers, and plotting the state on distant clients. 

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

You can run a simulation on your machine (using either the 2D or 3D envs):

```bash
python3 simulate.py  
```

Or host it on a server :

```bash
python3 server.py
```

Visualize it on distant clients : 

```bash
python3 client.py
```

And modify the state of the simulation using [this notebook client](notebook_controller.ipynb)

## TODO : 

- Stop the atrocious popping up of matplotlib interactive figures 
- Replace sending the first example state with pickle by generating it on the client side 
- Further check the interaction between 3D sim and notebook client 