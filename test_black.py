import numpy as np 

print( "I want to tes the black formatter"   )

def   function(arg1,arg2):

    result = arg1+arg2
    return result

import jax.numpy as jnp 

arg1 = jnp.array([0., 2., 3.])
arg2 = jnp.array([1.,6.,  28.])

function(arg1=arg1,arg2=arg2)