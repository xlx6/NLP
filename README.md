# NLP
nlp project

# Comparision of different methods

| model     | score |    netes          |
|--------------|--------|------------------|
| CountVectorizer + Random Forest | 0.84640  |  |
| Word2vec(Avg) + Random Forest | 0.83180  |  |
| Word2vec(Clustered) + Random Forest | 0.84172  |  |
| Word2vec(Avg) + MLP   | 0.87648  | |
| BERT + MLP   |   | freeze BERT, train MLP |
| BERT + MLP   | 0.92640  | unfreeze the last 4 layers of BERT|