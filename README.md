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

## 5. Evolução da ideia (fora do escopo da Entrega 1)

Proposta a discutir com o professor. A Entrega 1 continua sendo apenas o modo do desenvolvedor descrito acima.

### Modo 2: consultar um jogo existente

Hoje o usuário descreve um jogo que **ainda vai fazer**. No modo 2, qualquer jogador digita o **nome de um jogo já lançado** e recebe duas respostas.

**a) Desempenho real × esperado (usa o modelo).** O nome é buscado no próprio dataset, que já tem `AppID` e `Name`. As características do jogo passam pelo mesmo pipeline do modo 1, e a faixa prevista é comparada com a faixa real:

| Previsto | Real | Leitura |
|---|---|---|
| `media` | `alta` | superou o esperado para o perfil |
| `alta` | `baixa` | ficou abaixo do esperado para o perfil |
| igual | igual | desempenho dentro do esperado |

Essa parte reaproveita o modelo treinado e responde a uma pergunta nova: o sucesso veio do perfil do jogo ou de algo fora dele (marketing, qualidade, viralização)?

**b) Funil de conclusão pelas conquistas (consulta de dados, não é previsão).** A partir das conquistas, a resposta mostra quantos jogadores chegam a cada etapa do jogo e onde a maioria abandona. Exemplo:

| Conquista | Jogadores que obtiveram |
|---|---|
| Concluir o prólogo | 85% |
| Concluir o capítulo 1 | 61% |
| Concluir o capítulo 3 | 38% |
| Ver o final | 24% |

Leitura: 76% dos jogadores não chegaram ao final, e a maior queda acontece entre o prólogo e o capítulo 1.

### Viabilidade técnica do funil

| Etapa | Fonte | Situação |
|---|---|---|
| Nome → `AppID` | dataset do projeto | ✅ sem chamada externa |
| % de jogadores por conquista | Steam Web API `GetGlobalAchievementPercentagesForApp` | ✅ pública, não exige chave |
| Nome e descrição das conquistas | Steam Web API `GetSchemaForGame` | ✅ exige chave de API gratuita |
| Saber qual conquista é etapa da história e qual é o final | nenhuma | ⚠️ a Steam não marca isso |

O ponto difícil é o último. Para identificar as conquistas de progressão, há três caminhos, do mais simples ao mais caro:

1. **Heurística por texto:** palavras como "chapter", "act", "complete", "ending", "the end", "finish" no nome ou na descrição.
2. **Heurística por padrão:** conquistas de história costumam formar uma sequência com porcentagens decrescentes e costumam ser ocultas antes de desbloqueadas.
3. **Rotulagem manual** de algumas centenas de jogos, usada para validar as heurísticas.

### Limitações do funil

- Não se aplica a jogos sem conquistas nem a jogos sem final (multiplayer, roguelike, sandbox, simulação).
- Jogos com vários finais, ou com conquista de final perdível, distorcem a taxa de conclusão.
- A porcentagem da Steam conta donos que nunca abriram o jogo (bundles, jogos gratuitos, fins de semana grátis), o que faz a taxa de conclusão parecer menor do que é.
- A consulta é feita ao vivo na API da Steam: precisa de cache e de tratamento para quando a API não responder durante a demonstração.

### Vale a pena adicionar?

| Parte | Esforço | Recomendação |
|---|---|---|
| a) Real × esperado | baixo: reaproveita modelo e dataset | ✅ incluir na Entrega 4 como segundo endpoint da API |
| b) Funil de conclusão | médio: integração com a API da Steam + heurística | ⚠️ complemento opcional, depois da aplicação principal estar no ar |
| Prever a taxa de conclusão de um jogo novo | alto: exige rotular o final de mais de mil jogos | ❌ fora deste projeto; possível trabalho futuro |

O item (b) sozinho não é aprendizado de máquina, é consulta de dados. Por isso ele entra como **contexto** na resposta, e não como substituto do modelo. A parte avaliada do projeto continua sendo a previsão do modo 1 e a comparação do item (a).
