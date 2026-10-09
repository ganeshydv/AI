##

0. LLM: a function. Takes token IDs in, returns a probability for every possible next token
1. Tokenizer: text -> list of token IDs (integers), e.g. "Node.js is fast" -> [101, 2291, 1012, 1046, 2003, 3435, 102]. No neural network, just a fixed vocabulary
1. Token: a piece of text (a word, part of a word, or punctuation). Each token has one integer ID in the vocabulary
2. Embedding: a vector of numbers that represents a token (or a whole text). Token ID -> vector: [22, 43] = [[0.4, 0.1, 0.6, ..., N], [0.1, 0.6, 0.8, ..., N]]
2. Dimensions: how many numbers are in each vector (N = 384 for all-MiniLM-L6-v2)
2. Magnitude: length of a single vector, the Euclidean (L2) norm: sqrt(v1^2 + v2^2 + ... + vN^2)
2. Euclidean distance, cosine similarity, dot product: ways to compare two vectors (not magnitude)
3. Embedding matrix: table of vocab_size x dimensions (30,522 x 384). Row i = vector for token ID i
3. Transformer: stack of layers (attention + feed-forward) that turns each token's context-free vector into a contextual vector
4. Pooling: average the token vectors into one vector per text [tokens x 384] -> [384]
4. Normalization: divide a vector by its magnitude so its length is 1


##

- mulitiple dimensions
- 
```
text ---tokenizer--> tokens [numbers,..,]
 ---transformer-> Embedding matrix: vector for each token
 --> a. Add positsion vector
 --> b. Add Attention vector