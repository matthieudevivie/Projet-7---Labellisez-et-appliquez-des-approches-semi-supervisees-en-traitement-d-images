```mermaid
%%{init: {"themeVariables": {"fontFamily": "Arial, sans-serif"}, "flowchart": {"htmlLabels": false}}}%%
flowchart TD
    subgraph E1["Étape 1 — Nettoyage des données (01_exploration)"]
        A["Images brutes<br/>100 labellisées + 1406 sans label"] --> B["Suppression des doublons (MD5)<br/>96 fichiers retirés"]
    end
    B --> INV[("data/inventaire.csv<br/>99 labellisées + 1311 sans label")]

    INV --> SPLIT{"Split stratifié des 99 images labellisées<br/>une seule fois, random_state = 42"}
    SPLIT --> TRF["Train fort<br/>69 images (35 cancers / 34 normaux)"]
    SPLIT --> TEST["🔒 Test<br/>30 images (15 / 15)<br/>jamais utilisé pour entraîner ni pour choisir"]

    subgraph E2["Étape 2 — Prétraitement et features (02_features)"]
        P["Resize 224×224 → tenseur → normalisation ImageNet<br/>classe ImagesIRM (Dataset)"] --> L["Lots de 32 images<br/>DataLoader"]
        L --> R["ResNet-50 pré-entraîné, gelé<br/>fc remplacée par Identity"]
        EQ["Égalisation d'histogramme avant ResNet<br/>globale · avec masque · CLAHE"]
    end
    INV --> P
    R --> EMB[("data/embeddings.npy<br/>1410 × 2048")]
    P -. "variantes testées" .-> EQ
    EQ --> EMBV[("data/embeddings_globale · masque · clahe.npy")]
    EMBV -.-> CV

    subgraph E3["Étape 3 — Analyse non supervisée (03_clustering)"]
        S["Standardisation (StandardScaler)<br/>ajustée sans le test"] --> ACP["ACP et t-SNE en 2D : visualisation<br/>axe 1 = plan de coupe (biais confirmé)"]
        S --> NC["ACP pour le clustering<br/>20 / 50 / 100 composantes, 90 % de variance"]
        NC --> CL["5 méthodes, 2 groupes<br/>K-means · Ward · mélange gaussien · DBSCAN · HDBSCAN"]
        CL --> ARI["Choix par l'ARI sur le train (+ bootstrap)<br/>retenu : Ward, 100 composantes, ARI 0,58"]
        ARI --> VOTE["Nommage des clusters<br/>vote majoritaire du train fort"]
    end
    EMB --> S
    TRF -. "labels du train fort uniquement" .-> ARI
    TRF -. "vote" .-> VOTE
    VOTE --> LF[("data/labels_faibles.csv<br/>1311 images : 805 cancer / 506 normal<br/>justes à 88 % sur le train (83 % des cancers)")]

    subgraph E4["Étape 4 — Approche semi-supervisée (04_semi_supervise)"]
        H["Tête Linear(2048, 2) sur les embeddings<br/>ResNet gelé · Adam · réglages fixés à l'avance"]
        M2["② Faibles seuls<br/>1311 labels faibles · 10 epochs"] --> M3["③ Faibles puis forts<br/>② poursuivi sur le train fort · 50 epochs · lr ÷ 10"]
        M4["④ Forts seuls<br/>69 labels forts · 50 epochs"]
        H --> M2
        H --> M4
        CV["Validation croisée sur le train fort<br/>5 plis stratifiés × 3, mêmes plis pour tout<br/>arrêt anticipé sur la perte de validation"]
        CV --> GR["Grille de 27 configurations<br/>lr · L2 · tête linéaire ou 256 + dropout<br/>comparaison pli par pli"]
        GR --> TO["⑤ Tête optimisée<br/>256 neurones + dropout 0,5 · 8 epochs<br/>F2 0,938 en validation croisée (baseline 0,842)"]
        CV --> EQR["Égalisation : aucune variante<br/>ne fait mieux → non retenue"]
        LS["⑥ Label spreading<br/>graphe des 20 plus proches voisins (ACP 100)<br/>labels cachés retrouvés à 91 %"]
    end
    EMB --> H
    EMB --> CV
    EMB --> LS
    LF --> M2
    TRF --> M3
    TRF --> M4
    TRF --> CV
    TRF --> LS

    M2 --> EV["📊 Évaluation sur le MÊME test 🔒, une seule fois par approche<br/>rappel cancer, précision, F2, matrices de confusion<br/>10 graines pour les têtes (moyenne ± écart-type)"]
    M3 --> EV
    M4 --> EV
    TO --> EV
    LS --> EV
    TEST --> EV
    EV --> CMP["Résultats : ③ ≈ ④ ≈ ⑤ ≈ ⑥ ≈ 87 % de rappel, ② ≈ 75 %<br/>definition of done atteinte (rappel au-dessus de 60 %, précision ≥ 70 %)<br/>2 cancers axiaux ratés par toutes les méthodes : limite des features"]
    CMP --> REC["Recommandations : passage à 4 M images, budget 5 000 €"]
```
