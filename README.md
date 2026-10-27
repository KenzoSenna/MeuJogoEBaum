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
| Tamanho | _preencher com `df.shape`_ |

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
| `DummyClassifier(strategy='most_frequent')` | _preencher_ | _preencher_ | 0 |

O Dummy sempre responde a classe mais frequente. Por isso a precisão de `alta` é zero e o F1-macro fica baixo mesmo com acurácia alta. Esse é o número que o modelo da Entrega 3 precisa superar.
