# Revisão rápida para entrevista

## Projeto em uma frase

Usei dados de clientes de telecomunicações para prever `Churn`: se o cliente cancelou (`Yes`) ou não (`No`). Comparei Logistic Regression e Decision Tree e escolhi Logistic Regression como modelo final.

## Dados e variáveis

- **Target:** `Churn`.
- **Features:** informações demográficas, tempo como cliente, serviços contratados, tipo de contrato, forma de pagamento e cobranças.
- **`customerID`:** foi removida porque identifica cada cliente, mas não representa um comportamento útil para o modelo.
- **`TotalCharges`:** chegava como texto e alguns registros tinham espaços vazios. Converti para número com `errors="coerce"`, transformando valores inválidos em `NaN`.

## Treino e teste

- Separei 80% dos dados para treino e 20% para teste.
- Usei `random_state=42` para tornar a divisão reproduzível.
- Usei `stratify=y` para manter aproximadamente a mesma proporção de churn nos dois conjuntos.
- O teste ficou separado até a avaliação final.

## Pré-processamento

- `FunctionTransformer`: converte `TotalCharges` dentro da pipeline.
- `SimpleImputer(strategy="median")`: preenche valores numéricos ausentes usando uma mediana aprendida no treino.
- `StandardScaler`: coloca as variáveis numéricas em escalas comparáveis.
- `OneHotEncoder`: transforma categorias em colunas numéricas e ignora categorias novas na previsão.
- `ColumnTransformer`: aplica o tratamento correto a cada grupo de colunas.
- `Pipeline`: junta conversão, pré-processamento e modelo. Isso evita tratar treino e teste de formas diferentes e permite receber os dados crus.

## Modelos

- **Logistic Regression:** modelo linear de classificação escolhido como resultado final. Foi estável na validação e é simples de explicar.
- **Decision Tree sem controle:** obteve accuracy de treino muito alta e validação bem menor, sinal de overfitting.
- **Decision Tree com `max_depth=4`:** limitar a profundidade reduziu o overfitting, mas ela não substituiu a Logistic Regression como modelo final.
- **Overfitting:** acontece quando o modelo aprende detalhes demais do treino e não generaliza tão bem para dados novos.

## Métricas finais

- **Accuracy ≈ 80,55%:** proporção de todas as previsões que estavam corretas.
- **Recall ≈ 55,88%:** entre quem realmente cancelou, o modelo encontrou cerca de 56%.
- **Precision ≈ 65,72%:** entre quem o modelo marcou como churn, cerca de 66% realmente cancelaram.
- **F1 ≈ 60,40%:** equilíbrio entre Precision e Recall.

## Matriz de confusão

- **TN = 926:** não cancelou e o modelo previu não-churn.
- **FP = 109:** não cancelou, mas o modelo previu churn.
- **FN = 165:** cancelou, mas o modelo previu não-churn.
- **TP = 209:** cancelou e o modelo previu churn.

Os falsos negativos são a principal limitação porque representam clientes em risco que o modelo não identificou.

## Limitações e melhorias futuras

- O Recall ainda é limitado.
- Há uma quantidade relevante de falsos negativos.
- O modelo não está pronto para produção e as métricas precisam ser avaliadas junto aos custos do negócio.
- Como estudos futuros, eu poderia testar outros thresholds, novos modelos, novas features e custos diferentes para cada tipo de erro. Nada disso foi implementado nesta versão.
