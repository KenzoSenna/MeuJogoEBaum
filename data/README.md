# Dados

Fonte: [Steam Games Dataset — FronkonGames (Kaggle)](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset), licença MIT.

## `games.csv.gz` (versionado)

Cópia reduzida do dataset, com as mesmas 125.855 linhas e 40 colunas do original. O cabeçalho já vem corrigido (`Discount` e `DLC count` separados) e as colunas de texto livre e de URLs, que o modelo não usa, foram cortadas em 20 caracteres para o arquivo caber no repositório.

É o arquivo lido pelos notebooks `01_eda.ipynb` e `02_modelo.ipynb`. Não é preciso baixar nada para rodá-los.

## `games.csv` (não versionado)

Arquivo original do Kaggle, com cerca de 400 MB. Só o `00_primeiro_contato.ipynb` usa esse arquivo. Para rodar esse notebook de novo:

1. Acesse o link da fonte acima.
2. Baixe e extraia `games.csv` nesta pasta (`data/games.csv`).

Pela linha de comando, com a API do Kaggle configurada:

```bash
kaggle datasets download -d fronkongames/steam-games-dataset -p data/ --unzip
```
