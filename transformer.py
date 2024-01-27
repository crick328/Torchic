# parole > numeri tramite words embeddings
# positional encoding: prendo i valori della rete di word embeddings e gli do in input a delle
# funzioni seno e coseno che diventano sempre più larghe per ogni parola (tipo cos(output[0], sin(output[1]), cos(2*output[3]), ecc.)).
# Sommo gli output delle funzioni trigonometriche agli output della word embeddings per ottenere le parole fatte e finite.
# Ottenuti questi nuovi valori, faccio un'altra matmul con i weights, ma stavolta non sommo il bias e non uso la act_fun.
# Questi si chiamano Query Values. Si fa lo stesso processo per calcolare le Keys, valori che consentono di calcolare
# la similarità del significato della stessa parola in contesti diversi grazie al dot product