# Project_music_generation

Dessa filer behöver följande moduler för att fungera korrekt, modulerna måste instaleras korrekt också:



Sytax för demo: 
*python demo.py rate checkpoint1 chockpoint2....checkpointN*
Måste ha rate i hz, rekomenderas 8000. Ingen checkpoint behövs skrivas in men då kommer den välja slump varje generation mellan alla checkpoints. Alla checkpoint nummer man skriver in så kommer den slump mellan dem.

Syntax för create_dataset:
*python create_dataset.py inputDir outputDir*
input dir ska vara en mapp med wave filer som behöver bli converterade förhållande till dennas mapp. outpurdir är var det nya datasettet kommer sparas förhållande till dennas mapp.

Syntax för training:
*python training.py dataSetDir batchSize nrEpoch*
dataSetDir är var alla bilder som den ska träna på finns förhållande till denna mappen. batchSize är batchsize. nrEpoch är hur många gåner den ska gå igenom alla bilder, var 10 epoch så sparar den en checkpoint.
