# Marketing bancário: previsão de adesão a depósito a prazo

Classificador para **priorizar ligações de telemarketing** de um banco, prevendo quais clientes têm maior propensão a aderir a um depósito a prazo. Os dois modelos — **regressão logística** e **random forest** — e tudo o que os cerca (descida do gradiente, árvores CART, *bootstrap* balanceado, otimização bayesiana, ROC/AUC/F1) foram **implementados do zero em R base**, sem bibliotecas de modelagem.

## O problema de negócio

Toda campanha de telemarketing tem custo por ligação e equipe limitada. A base [*Bank Marketing*, UCI]; Moro et al., 2014) traz **41.188 contatos** reais de um banco português, descritos por perfil do cliente, histórico de contato e contexto macroeconômico. O alvo é fortemente desbalanceado: só **12,7%** dos clientes aderem. A pergunta: *de toda a base, em quem vale a pena gastar uma ligação?*

A variável `duration` (duração da ligação) é descartada por ser **vazamento** — só é conhecida depois do contato. O modelo precisa decidir *antes* de ligar.

## Resultados

Modelo final: **random forest** (*Balanced RF*, 17 variáveis selecionadas, corte 0,50), avaliada **uma única vez** num conjunto de teste reservado.

| Métrica | Teste |
|---|---:|
| Sensibilidade (recupera aderentes) | **0,575** |
| F1 (classe "sim") | **0,503** |
| AUC (ROC) | 0,803 |
| *Average Precision* | 0,481 (~3,8× a base) |
| Acurácia | 0,856 |

Sem sinal de *overfitting*: as diferenças entre a validação OOB (treino) e o teste são **todas ≤ 0,02**, e a AUC é idêntica (0,803). Na prática, o modelo recupera **~58% dos clientes que de fato aderem**, concentrando-os no topo da lista de prioridade.


## Etapas do trabalho

1. **Limpeza** — remoção de duplicatas e de registros `unknown` (41.188 → 30.478).
2. **Análise exploratória** — alvo desbalanceado, distribuições, sinal nos indicadores macroeconômicos e multicolinearidade.
3. **Transformação** — descarte de `duration`, recodificação de `pdays`, *one-hot* (56 colunas) e partição estratificada 70/30 com padronização ajustada no treino.
4. **Seleção por correlação** — relevância (`|r| ≥ 0,05`) + remoção de redundância (`|cor| > 0,80`): 56 → 17 colunas.
5. **Regressão logística** — sigmoide + log-loss + descida do gradiente.
6. **Desbalanceamento** — *random undersampling* 2:1 (só no treino).
7. **Random forest** — Gini + CART + *bootstrap* balanceado + erro OOB.
8. **Impacto da seleção** — 17 vs 56 variáveis.
9. **Tuning** — otimização bayesiana (processo gaussiano + *Expected Improvement*) com F1 por validação cruzada.
10. **Avaliação final** — teste reservado, ROC, *precision-recall* e importância por permutação.

## Tecnologias

**R base** (modelagem, métricas e gráficos, sem pacotes de *machine learning*). O notebook do relatório é escrito inteiramente em **R** e contém o código completo do projeto, com as figuras embutidas como saída de cada célula.

## Referências

- Moro, S.; Cortez, P.; Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing*. **Decision Support Systems**, 62, 22–31.
- Dua, D.; Graff, C. *UCI Machine Learning Repository — Bank Marketing Data Set*. University of California, Irvine.
