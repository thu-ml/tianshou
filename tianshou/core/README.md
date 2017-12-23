#TODO: 

Separate actor and critic. (Important, we need to focus on that recently)

# policy

YongRen

### base, stochastic

follow OnehotCategorical to write Gaussian, can be in the same file as stochastic.py

### deterministic

not sure how to write, but should at least have act() method to interact with environment

referencing QValuePolicy in base.py, should have at least the listed methods.


# losses

TongzhengRen

seems to be direct python functions. Though the management of placeholders may require some discussion. also may write it in a functional form.

# policy, value_function

naming should be reconsidered. Perhaps use plural forms for all nouns