import pickle
counter = pickle.load( open( "count_vect.p", "rb" ) )
training_counts =  pickle.load( open( "train.p", "rb" ) )