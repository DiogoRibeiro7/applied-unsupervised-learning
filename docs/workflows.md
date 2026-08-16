# Modelling Workflows

Diagrams of the workflows used across the project. They render natively on GitHub
(Mermaid). The first is the shared backbone; the rest specialise it per family.

## The shared unsupervised workflow

```mermaid
flowchart TD
    A[Frame the problem<br/>without labels] --> B[Generate / load data<br/>deterministic seeds]
    B --> C[Feature engineering<br/>and scaling]
    C --> D[Fit several methods<br/>with different assumptions]
    D --> E[Evaluate without labels<br/>internal metrics + stability]
    E --> F{Structure robust?}
    F -- no --> D
    F -- yes --> G[Interpret and name<br/>the structure]
    G --> H[State limitations<br/>and next steps]
```

## Clustering and segmentation (notebooks 01, 06, 07)

```mermaid
flowchart LR
    F[Features] --> S[Scale]
    S --> M[KMeans / GMM /<br/>Agglomerative / DBSCAN]
    M --> V[Internal metrics<br/>silhouette, DB, CH]
    M --> ST[Stability<br/>bootstrap + AMI]
    V --> SEL[Select k / method]
    ST --> SEL
    SEL --> C[Consensus clustering<br/>reduce variance]
    C --> P[Profile + name segments]
```

## Anomaly detection (notebook 03)

```mermaid
flowchart LR
    E[Event stream] --> W[Scale sensor features]
    W --> SC[Score: IsolationForest / LOF /<br/>robust cov / one-class SVM / PCA error]
    SC --> RK[Rank-normalise<br/>then combine]
    RK --> T[Rank top-k]
    T --> EX[Review top anomalies]
    EX --> EV[Offline check:<br/>precision@k on hidden labels]
```

## Topic modelling (notebook 04)

```mermaid
flowchart LR
    D[Documents] --> CL[Clean text]
    CL --> TF[TF-IDF]
    TF --> NMF[NMF / LSA]
    NMF --> TT[Topic-term tables]
    TT --> LB[Auto labels]
    NMF --> AS[Assign documents<br/>+ confidence]
    TT --> Q[Coherence + diversity]
    Q --> FM[Check failure modes]
```

## Recommender embeddings (notebook 05)

```mermaid
flowchart LR
    L[Interaction log] --> SP[Sparse matrix]
    SP --> MF[TruncatedSVD / NMF]
    MF --> UE[User embeddings]
    MF --> IE[Item embeddings]
    IE --> SI[Similar items]
    IE --> PG[Product groups]
    UE --> RC[Recommendations<br/>mask seen items]
```

## Streaming and drift (notebook 08)

```mermaid
flowchart LR
    B[Batch t] --> U[partial_fit<br/>mini-batch KMeans]
    U --> CS[Centroid shift]
    U --> PSI[PSI vs reference]
    CS --> M{Drift?}
    PSI --> M
    M -- yes --> R[Flag / re-fit]
    M -- no --> B2[Batch t+1]
    R --> B2
```

## Production path (CLI / API)

```mermaid
flowchart LR
    CFG[YAML config] --> TR[Train via CLI]
    TR --> ART[Artifact + metadata]
    TR --> RPT[JSON report]
    TR --> TRK[Experiment log JSONL]
    ART --> API[FastAPI service]
    ART --> BS[Batch score CSV]
    API --> EP[/cluster/assign<br/>/anomaly/score/]
```
