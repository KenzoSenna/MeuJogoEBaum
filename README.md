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
| Tamanho | `(100292, 40)` |

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
- Removidos jogos lançados nos últimos 12 meses: ainda não acumularam donos e cairiam em `baixa` injustamente.

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
