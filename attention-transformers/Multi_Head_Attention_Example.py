import numpy as np
from scipy.special import softmax

print("Step 1: Input : 3 inputs, d_model=4")
x = np.array([[1.0, 0.0, 1.0, 0.0],
              [0.0, 2.0, 0.0, 2.0],
              [1.0, 1.0, 1.0, 1.0]])
print("x:",x)

print("Step 2: weights 3 dimensions x d_model=4")
w_query = np.array([[1, 0, 1],
                    [1, 0, 0],
                    [0, 0, 1],
                    [0, 1, 1]])
print("w_query:",w_query)

w_key = np.array([[0, 0, 1],
                  [1, 1, 0],
                  [0, 1, 0],
                  [1, 1, 0]])
print("w_key:",w_key)

w_value = np.array([[0, 2, 0],
                    [0, 3, 0],
                    [1, 0, 3],
                    [1, 1, 0]])
print("w_value:",w_value)

print("Step 3: Matrix multiplication to obtain Q,K,V")
print("Query: x * w_query")
Q = np.matmul(x,w_query)
print("Q:",Q)

print("Key: x * w_key")
K = np.matmul(x,w_key)
print("K:",K)

print("Value: x * w_value")
V = np.matmul(x,w_value)
print("V:",V)

print("Step 4: Scaled Attention Scores")
k_d = 1 #Equation is normally the square root of the number of dimensions (3 in this case)
attention_scores = (Q @ K.transpose())/k_d
print(attention_scores)

print("Step 5: Scaled softmax attention_scores for each vector")
attention_scores[0] = softmax(attention_scores[0])
attention_scores[1] = softmax(attention_scores[1])
attention_scores[2] = softmax(attention_scores[2])
print(attention_scores[0])
print(attention_scores[1])
print(attention_scores[2])

print("Step 6: attention value obtained by score1/k_d * V")
print(V[0])
print(V[1])
print(V[2])
attention1 = attention_scores[0].reshape(-1,1)
attention1 = attention_scores[0][0]*V[0]
print("Attention 1:",attention1)

attention2 = attention_scores[0][1]*V[1]
print("Attention 2:",attention2)

attention3 = attention_scores[0][2]*V[2]
print("Attention 3:",attention3)

print("Step 7: summed the results to create the first line of the output matrix")
attention_input1 = attention1 + attention2 + attention3
print(attention_input1)

print("Step 8: Step 1 to 7 for inputs 1 to 3")
#This is assuming that we had actually gone through the whole process for all 3
#We'll just take a random matrix of the correct dimensions in lieu
attention_head1 = np.random.random((3, 64))
print(attention_head1)

print("Step 9: We assume we have trained the 8 heads of the attention sub-layer")
z0h1 = np.random.random((3,64))
z1h2 = np.random.random((3,64))
z2h3 = np.random.random((3,64))
z3h4 = np.random.random((3,64))
z4h5 = np.random.random((3,64))
z5h6 = np.random.random((3,64))
z6h7 = np.random.random((3,64))
z7h8 = np.random.random((3,64))
print("shape of one head",z0h1.shape,"dimension of 8 heads",64*8)

print("Step 10: Concatenation of heads 1 to 8 to obtain the original 8x64=512 output dimension of the model")
output_attention = np.hstack((z0h1, z1h2, z2h3, z3h4, z4h5, z5h6, z6h7, z7h8))
print(output_attention)

