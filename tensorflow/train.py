# implement train logic and dataloading

# use model.py to generate the RNN object, which implements all the needed components and returns a session
import tensorflow as tf
import model

mo = model.Rnn(model_name="model_name", summary_name="summary_name")
session = mo.build(layers=7, 
                   filters=128,
                   state_depth=128, 
                   seqlen=10, 
                   m_frames=1, 
                   p_frames=1, 
                   in_ch=3)

