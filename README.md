# Meu Jogo É Baum?

**Integrantes:** Kenzo Senna, Hudson Batista Brandão, Matheus Felipe

## 1. Problema

> Dado o gênero, as tags, o preço, as plataformas, os idiomas, o mês de lançamento e o número de conquistas de um jogo, prever a faixa de relevância que ele tende a atingir na Steam, para que um desenvolvedor independente possa ajustar o posicionamento do jogo antes de lançá-lo.

- **Usuário:** desenvolvedor indie ou pequeno estúdio, antes do lançamento.
- **Tipo:** classificação com 3 classes.
- **Alvo:** coluna `Estimated owners` (donos estimados pelo SteamSpy), agrupada assim:

| Classe | Donos estimados |
|---|---|
| `baixa` | menos de 20 mil |
| `media` | 20 mil a 200 mil |
| `alta` | 200 mil ou mais |

O alvo já existe no dataset. As faixas do SteamSpy só são agrupadas em três, sem criar rótulo novo.

## 2. Dados

| Item | Valor |
|---|---|
| Fonte | [Steam Games Dataset — FronkonGames (Kaggle)](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset) |
| Origem | API oficial da Steam + SteamSpy |
| Licença | MIT (uso acadêmico permitido) |
| Tamanho | 125.855 linhas × 40 colunas no arquivo; 100.292 linhas após os filtros |

**Por que este dataset:** tem mais de mil linhas, a coluna-alvo já existe, não está na lista de datasets proibidos e trata de um problema real de quem desenvolve jogos.

### Colunas de entrada

Todas passam no teste do formulário: o desenvolvedor sabe cada uma antes de lançar.

| Coluna | Campo no formulário |
|---|---|
| `Genres` | gênero |
| `Tags` | tags |
| `Categories` | modos de jogo (single, multi, co-op) |
| `Price` | preço (USD) |
| `Windows`, `Mac`, `Linux` | plataformas |
| `Supported languages` | idiomas |
| `Release date` | mês de lançamento |
| `Required age` | classificação etária |
| `Achievements` | nº de conquistas |

### Colunas excluídas

`Positive`, `Negative`, `Recommendations`, `Peak CCU`, `User score`, `Metacritic score` e os tempos de jogo **só existem depois do lançamento**. Usá-las seria vazar o alvo. `AppID`, `Name` e URLs foram excluídas porque não descrevem o jogo.

### Filtros

- Removidos jogos com `Estimated owners = "0 - 0"`: o SteamSpy não tem o dado, não significa zero donos.
- Removidos jogos lançados a partir de 01/12/2025: ainda não acumularam donos e cairiam em `baixa` injustamente. A proposta previa cortar 12 meses, mas a coleta do dataset vai até março de 2026, então o corte ficou em cerca de 3 meses (ver limitações na seção 6).

## 3. Métrica

**Principal: F1-macro.** A maioria dos jogos da Steam fica abaixo de 20 mil donos. Um modelo que sempre responde `baixa` teria acurácia alta sem aprender nada. O F1-macro dá o mesmo peso às três classes e expõe esse modelo.

**Secundária: precisão da classe `alta`.** O erro mais caro é dizer "alta" para um jogo que não vai chegar lá, porque isso gera expectativa falsa e pode levar a investimento errado.

## 4. Baseline

Divisão 80/20 estratificada (`random_state=42`). O teste fica guardado até a Entrega 3. O baseline é medido com validação cruzada de 5 dobras, só no treino.

| Modelo | F1-macro | Acurácia | Precisão `alta` |
|---|---|---|---|
| `DummyClassifier(strategy='most_frequent')` | 0.2847 | 0.7453 | 0 |

O Dummy sempre responde a classe mais frequente. Por isso a precisão de `alta` é zero e o F1-macro fica baixo mesmo com acurácia alta. Esse é o número que o modelo da Entrega 3 precisa superar.

## 5. Tabela de decisões (Entrega 2)

Toda a EDA foi feita no notebook `notebooks/01_eda.ipynb` olhando apenas para o treino (`X_tr` e `y_tr`).

| Coluna | Tipo | Achado na EDA | Decisão | Onde |
|---|---|---|---|---|
| `Price` | numérica | assimétrica à direita; 7 valores acima de 200 | > 200 vira NaN; mediana + StandardScaler | limpar + pipeline |
| `Achievements` | numérica | 172 valores acima de 500 (spam); assimétrica | > 500 vira NaN; mediana + StandardScaler | limpar + pipeline |
| `Supported languages` | numérica | 72 linhas com `[]`; mediana sobe na classe alta | `[]` vira NaN; contar idiomas; mediana + StandardScaler | limpar + pipeline |
| `Genres` | nominal | 27 gêneros; 0,13% vazia | pegar 1º gênero; moda + one-hot, ignore | limpar + pipeline |
| `Tags` | nominal | 16,94% vazia; muitas tags com poucos jogos | pegar 1ª tag; moda + indicador + one-hot, ignore | limpar + pipeline |
| `Categories` | nominal | 0,82% vazia; single-player é maioria | pegar modo principal; moda + one-hot, ignore | limpar + pipeline |
| `Release date` | data | data em texto (`Oct 23, 2025`) | extrair mês (1 a 12); moda + one-hot, ignore | limpar + pipeline |
| `Required age` | ordinal | 98,8% livre (0); ordem natural por idade | agrupar em faixas (`Livre` a `18+`); ordinal com ordem fixa | limpar + pipeline |
| `Windows`, `Mac`, `Linux` | binária | sem vazios; maior correlação é 0,55 (< 0,9) | manter todas; one-hot | limpar + pipeline |
| `AppID`, `Name`, URLs, `About the game`, `Reviews`, `Notes`, `Developers`, `Publishers`, `Discount`, `DLC count` | ID / texto livre | identificadores ou textos fora do formulário | remover | limpar(df) |
| `Positive`, `Negative`, `Recommendations`, `Peak CCU`, `User score`, `Metacritic score`, `Score rank`, tempos de jogo | vaza | só existem depois do lançamento | remover | limpar(df) |

No teste de sanidade com validação cruzada (`cv=5`), o `Pipeline` com `ColumnTransformer` e `LogisticRegression(max_iter=1000)` rodou sem `NaN`, manteve as `80233` linhas de treino e atingiu **F1-macro = 0.4147** (superando o `0.2847` do baseline da Entrega 1).

## 6. Modelo e avaliação (Entrega 3)

Toda a modelagem está no notebook `notebooks/02_modelo.ipynb`, que usa a mesma base, a mesma `limpar(df)`, a mesma divisão (80.233 jogos no treino e 20.059 no teste) e o mesmo `ColumnTransformer` da Entrega 2. As comparações foram feitas por validação cruzada de 5 dobras no treino. O teste foi aberto uma única vez, depois de o modelo estar escolhido.

### Comparação dos algoritmos (validação cruzada no treino)

| Algoritmo | Peso das classes | F1-macro | Precisão `alta` | Recall `alta` |
|---|---|---|---|---|
| Dummy (`most_frequent`) | sem peso | 0.2847 | 0 | 0 |
| LogisticRegression | sem peso | 0.4146 | 0.5589 | 0.1750 |
| LogisticRegression | moderado (1-2-3) | 0.5010 | 0.4155 | 0.2984 |
| LogisticRegression | balanced | 0.4511 | 0.1980 | 0.6510 |
| RandomForest | sem peso | 0.4356 | 0.6577 | 0.1946 |
| RandomForest | moderado (1-2-3) | 0.5344 | 0.4903 | 0.3579 |
| RandomForest | balanced | 0.4855 | 0.2502 | 0.6556 |
| HistGradientBoosting | sem peso | 0.4787 | 0.5628 | 0.2912 |
| **HistGradientBoosting** | **moderado (1-2-3)** | **0.5421** | **0.4579** | **0.4020** |
| HistGradientBoosting | balanced | 0.4885 | 0.2543 | 0.6470 |

No peso moderado, um jogo `media` vale 2 e um jogo `alta` vale 3 durante o treino. O Dummy foi medido de novo nessa divisão e deu o mesmo F1-macro da Entrega 1.

### Decisões

| Pergunta | Achado | Decisão |
|---|---|---|
| Qual peso para as classes? | sem peso, o modelo quase não prevê `media` e `alta`; com `balanced`, a precisão da `alta` cai para 0.20 a 0.25 | peso moderado (1-2-3), que dá o melhor F1-macro nos três algoritmos |
| Qual algoritmo? | HistGradientBoosting 0.5421, RandomForest 0.5344, LogisticRegression 0.5010 | `HistGradientBoostingClassifier` |
| Vale ajustar os hiperparâmetros? | em uma grade de 9 combinações, a melhor ganha do padrão por 0.0005, menos que o desvio entre dobras (0.0019) | manter o padrão do scikit-learn |

### Resultado no teste

| Modelo | F1-macro | Acurácia | Precisão `alta` | Recall `alta` |
|---|---|---|---|---|
| `DummyClassifier(strategy='most_frequent')` | 0.2847 | 0.7452 | 0 | 0 |
| HistGradientBoosting, pesos 1-2-3 | 0.5554 | 0.7423 | 0.4786 | 0.4384 |

O modelo quase dobra o F1-macro do baseline. A acurácia fica igual à do Dummy, o que confirma que ela não serve como métrica neste problema.

| Classe | Precisão | Recall | F1 | Jogos no teste |
|---|---|---|---|---|
| `baixa` | 0.8285 | 0.8780 | 0.8525 | 14.948 |
| `media` | 0.4001 | 0.3206 | 0.3560 | 4.039 |
| `alta` | 0.4786 | 0.4384 | 0.4576 | 1.072 |

### Análise de erros

Matriz de confusão no teste:

| | Prevista `baixa` | Prevista `media` | Prevista `alta` |
|---|---|---|---|
| **Real `baixa`** | 13.124 | 1.618 | 206 |
| **Real `media`** | 2.438 | 1.295 | 306 |
| **Real `alta`** | 278 | 324 | 470 |

- **Erros entre faixas vizinhas:** 4.686 dos 5.170 erros, nove em cada dez. O mais comum é prever `baixa` para um jogo `media` (60,4% da classe `media`).
- **Erros graves, entre `baixa` e `alta`:** 484 jogos, 2,4% do teste.
- **Previsões de `alta`:** de 982, 470 estavam certas, 306 eram `media` e 206 eram `baixa`.
- **Confiança:** a taxa de acerto sobe com a probabilidade devolvida pelo modelo: 46,4% quando ela é de até 50% e 86,1% quando fica entre 70% e 80%. A aplicação vai avisar quando a confiança for de até 50%.
- **Colunas que mais pesam:** `Tags`, `Supported languages` e `Achievements`.

### Limitações encontradas

- **Jogos sem tag.** 16,9% dos jogos do teste não têm tag, e 99,9% deles são `baixa`. As tags são votadas pelos jogadores depois do lançamento, então a falta de tag é consequência da baixa popularidade. No formulário sempre haverá uma tag. Considerando só os jogos com tag, o F1-macro no teste é 0.5417 (Dummy: 0.2730) e a acurácia é 0.6899. Próximo passo: tirar do treino os jogos sem tag.
- **Filtro de lançamentos recentes.** O corte está em 01/12/2025, mas a coleta vai até março de 2026. Os jogos de 2025 entram no treino e 91,8% deles são `baixa`, em parte por serem recentes. Próximo passo: cortar a partir de março de 2025 e refazer a EDA e a modelagem.
- **Tempo de mercado.** O alvo é o total acumulado de donos, e o formulário não tem o ano de lançamento. O modelo subestima jogos antigos: entre os lançados até 2016, 35,9% são `baixa`, mas o modelo prevê `baixa` para 62,6%.
- **Associação, não causa.** Mais idiomas e mais conquistas aparecem em jogos maiores. Mudar esses campos não garante mais donos.

### Artefato

| Arquivo | Conteúdo |
|---|---|
| `models/pipeline.joblib` | `Pipeline` completo (pré-processador + modelo) |
| `models/modelo_info.json` | nome e versão do modelo, colunas de entrada, exemplo de entrada, métricas e versões das bibliotecas |

O pipeline espera as 11 colunas no formato que a `limpar(df)` produz e precisa ser carregado com `scikit-learn==1.9.1`, versão fixada no `requirements.txt`.

## 7. Como reproduzir

```bash
pip install -r requirements.txt
jupyter notebook notebooks/02_modelo.ipynb
```

O `data/games.csv.gz` já está no repositório. O notebook leva cerca de 10 minutos e regrava os arquivos da pasta `models/`.
