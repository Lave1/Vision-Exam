# Vision-Exam

## Cartella immagini

#### Dentro la cartella immagini ci sono tante cartelle tante le diverse immagini che si sono scaricate con il file image_downloader.

## Funziononamento file image_downloader.ipynb

#### il file image_downloader.ipynb funzione che basta cambiare nella query = "testo_da_cambiare" inserirendo il nome delle immagini che vogliamo scaricare (es: query = "macchina") e quante in num_images = "" (es: num_images = "10000") aggiunti questi campi il codice crea delle cartelle chiamate con il nome della query contenenti le n immagini messe in num_images, I nomi delle cartelle verrano usati per creare le classi.

## Funzionamnto del model.ipynb

#### **Il modello funziona**

1. **Primo strato convoluzionale (Conv2D):** Questo strato estrae 32 caratteristiche utilizzando filtri di dimensione 3x3. L'attivazione ReLU viene applicata per introdurre non linearità.
2. **Primo strato di max pooling:** Riduce la dimensione spaziale dell'output della convoluzione.
3. **Secondo e Terzo strato convoluzionale:** Aggiunge ulteriori strati convoluzionali per estrarre feature più complesse.
4. **Strato di flatten:** Converte l'output bidimensionale dei livelli convoluzionali in un vettore unidimensionale.
5. **Primo strato denso (Dense):** Aggiunge un layer completamente connesso con 64 neuroni e attivazione ReLU.
6. **Secondo strato densamente connesso con attivazione softmax:** L'ultimo layer con 9 neuroni (uno per ogni classe) e attivazione softmax per la classificazione multiclasse.

Il modello viene quindi compilato con l'ottimizzatore Adam, la funzione di loss categorical_crossentropy e misurato con l'accuratezza.
