# Aula prática — Machine Learning e vieses: do zero ao teste de modelos

>**Ambiente:** Google Colab. 
**Arquivos:** `clientes_ml_etica.csv` e `pratica_ml_etica.ipynb`. Esta prática amplia o roteiro de *Vieses em Dados e Análise Justa* com uma base utilizável e dois algoritmos.

## 1. O problema que vamos resolver

Uma loja fictícia quer prever se um cliente realizou uma compra (`comprou`: 1 = sim; 0 = não) a partir de `idade`, `renda` e `visitas_site`. Os dados são **sintéticos**: não representam pessoas reais nem autorizam conclusões sobre grupos reais. Para a discussão ética, adicionamos `grupo` (A ou B) e `recebeu_oferta` (0 ou 1). **O grupo é uma categoria inventada, sem significado demográfico**; na simulação, o grupo A recebeu ofertas com maior frequência, o que pode afetar as compras observadas.

### Dicionário de dados

| Coluna | Tipo | Significado | Usada inicialmente para prever? |
|---|---|---|---|
| `idade` | Numérica inteira | Idade do cliente | Sim |
| `renda` | Numérica | Renda mensal fictícia em reais | Sim |
| `visitas_site` | Numérica inteira | Número de visitas ao site | Sim |
| `grupo` | Categórica | Grupo fictício A ou B | Não; usada na auditoria |
| `recebeu_oferta` | Binária | 1 = recebeu oferta; 0 = não | Não inicialmente; analisada depois |
| `comprou` | Binária | 1 = comprou; 0 = não | **Não: é o alvo (`y`)** |

## 2. Antes do código: o vocabulário essencial

- **Machine Learning (aprendizado de máquina):** técnicas para ajustar modelos a exemplos, buscando padrões úteis para previsões.
- **Algoritmo:** procedimento de aprendizado escolhido, por exemplo, regressão logística ou árvore de decisão.
- **Modelo treinado:** resultado de aplicar o algoritmo a um conjunto de dados; contém parâmetros ou regras aprendidas.
- **Classificação:** previsão de uma categoria, aqui `0` ou `1`. Não é regressão numérica contínua.
- **Feature / variável de entrada (`X`):** informação fornecida ao modelo para prever, como renda.
- **Target / variável-alvo (`y`):** resposta conhecida que o modelo tenta aprender a prever, aqui `comprou`.
- **Treinamento:** fase em que o modelo ajusta seus parâmetros a exemplos com respostas.
- **Teste:** fase em que avaliamos o modelo em exemplos que ele não viu durante o treinamento.
- **Previsão (`predict`):** categoria escolhida pelo modelo.
- **Probabilidade (`predict_proba`):** estimativa numérica da probabilidade atribuída a cada categoria; não equivale a uma garantia de que a estimativa esteja bem calibrada.

### Regressão logística: o que é?

Apesar do nome, `LogisticRegression` é frequentemente usada para **classificação binária**. O algoritmo combina numericamente as características de entrada e transforma o resultado em uma probabilidade com a função logística (sigmoide). Em geral, `predict()` seleciona a classe mais provável. Exemplo **ilustrativo**: se o modelo estima 0,73 para a classe `1`, prevê compra. A regressão logística costuma ser um bom ponto de partida, especialmente quando a relação pode ser aproximada por uma fronteira de decisão linear no espaço transformado das características. Para variáveis em escalas muito diferentes (idade, renda), usamos `StandardScaler` no pipeline.

### Árvore de decisão: o que é?

`DecisionTreeClassifier` constrói uma sequência de perguntas sobre as características, como “`visitas_site <= 3`?” e “`renda <= 4500`?”. Cada divisão tenta separar exemplos de classes diferentes. O caminho percorrido por um registro termina em uma folha com uma previsão. A árvore é fácil de visualizar, mas uma árvore muito profunda pode **memorizar o treino** (sobreajuste ou *overfitting*). Usamos `max_depth=3` e `min_samples_leaf=15` para limitar a complexidade.

**Diferença essencial:** a regressão logística aprende coeficientes numéricos para combinar entradas; a árvore aprende condições de divisão. Nenhum deles é automaticamente justo ou imparcial.

## 3. Abrindo o Google Colab

1. Entre em https://colab.research.google.com/ e crie um **Novo notebook**.
2. Faça download de `clientes_ml_etica.csv` e `pratica_ml_etica.ipynb`.
3. No Colab, use **Arquivo → Fazer upload de notebook** e selecione o `.ipynb` (ou copie os blocos abaixo em células).
4. No painel lateral esquerdo, clique no ícone de **pasta** e depois em **Fazer upload** para enviar `clientes_ml_etica.csv`.
5. Execute as células na ordem usando o botão ▶. O arquivo enviado fica no ambiente temporário do Colab e pode precisar ser reenviado em outra sessão.
6. Se faltar alguma biblioteca, execute `!pip install pandas scikit-learn matplotlib` em uma célula antes dos imports. No Colab, elas normalmente já estão disponíveis.

## 4. Carregamento e exploração inicial

```python
import pandas as pd

df = pd.read_csv('clientes_ml_etica.csv')
print(df.head())       # primeiras linhas
print(df.shape)        # (linhas, colunas)
print(df.info())       # tipos e valores não nulos
print(df.isna().sum()) # valores ausentes
print(df['comprou'].value_counts())
```

**Explique:** `pd.read_csv` transforma o CSV em um DataFrame; `head()` mostra amostra visual; `shape` mostra dimensões; `value_counts()` revela quantas compras e não compras existem. Antes de treinar, verificamos se há valores faltantes e como as classes estão distribuídas.

## 5. Separando entradas e resposta

```python
from sklearn.model_selection import train_test_split

colunas_entrada = ['idade', 'renda', 'visitas_site']
X = df[colunas_entrada]
y = df['comprou']
grupos = df['grupo']

X_treino, X_teste, y_treino, y_teste, grupo_treino, grupo_teste = train_test_split(
    X, y, grupos, test_size=0.25, random_state=42, stratify=y
)
print('Treino:', len(X_treino), 'Teste:', len(X_teste))
```

**Explique cada argumento:** `test_size=0.25` reserva 25% para teste; `random_state=42` permite repetir a mesma divisão; `stratify=y` tenta preservar a proporção das classes 0 e 1 nos dois conjuntos. Passamos `grupos` na mesma divisão para manter cada grupo alinhado ao cliente correto, mas **não usamos o grupo como entrada do modelo**.

**Por que não usar `modelo.predict(X)` logo depois de `fit(X, y)`?** Isso avalia o modelo nas mesmas linhas que ele viu durante o treino. Um resultado alto nesse cenário pode ser apenas memorização, e não capacidade de prever casos novos.

## 6. Modelo 1 — Regressão logística

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

modelo_logistico = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)
modelo_logistico.fit(X_treino, y_treino)
prev_logistica = modelo_logistico.predict(X_teste)
prob_logistica = modelo_logistico.predict_proba(X_teste)[:, 1]
```

**Linha a linha:** `StandardScaler` centraliza e ajusta a escala das entradas usando apenas o conjunto de treinamento; `make_pipeline` encadeia a transformação e o modelo, aplicando a mesma transformação ao teste; `LogisticRegression` define o algoritmo; `fit()` aprende; `predict()` produz 0 ou 1; `predict_proba()[:, 1]` seleciona a probabilidade estimada para a classe 1. `max_iter=1000` permite mais iterações de otimização se necessário.

## 7. Modelo 2 — Árvore de decisão

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

modelo_arvore = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=15, random_state=42
)
modelo_arvore.fit(X_treino, y_treino)
prev_arvore = modelo_arvore.predict(X_teste)

plt.figure(figsize=(15, 7))
plot_tree(modelo_arvore, feature_names=colunas_entrada,
          class_names=['Não comprou', 'Comprou'], filled=True, rounded=True)
plt.show()
```

**Como ler a árvore:** comece no topo; compare o valor da característica com a condição escrita no nó; siga o ramo correspondente; repita até chegar à folha. `max_depth=3` limita a profundidade, e `min_samples_leaf=15` impede folhas com menos de 15 amostras do treino. A cor ajuda a visualizar a classe predominante; não indica se a decisão é justa.

## 8. Avaliando os dois modelos

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

for nome, previsoes in [('Logística', prev_logistica), ('Árvore', prev_arvore)]:
    print('\n', nome)
    print('Acurácia:', round(accuracy_score(y_teste, previsoes), 3))
    print('Matriz de confusão:\n', confusion_matrix(y_teste, previsoes, labels=[0, 1]))
    print(classification_report(y_teste, previsoes, zero_division=0))
```

**Acurácia:** proporção de previsões corretas entre todas as previsões. **Matriz de confusão** (linhas = real; colunas = previsto, na ordem 0, 1): canto superior esquerdo = verdadeiro negativo; superior direito = falso positivo; inferior esquerdo = falso negativo; inferior direito = verdadeiro positivo. **Precisão (precision) da classe 1:** entre as compras previstas, quantas realmente ocorreram? **Revocação/sensibilidade (recall/TPR) da classe 1:** entre as compras reais, quantas foram identificadas? **F1:** combina precisão e revocação. Se quase todos fossem não compradores, um modelo que previsse sempre 0 poderia ter acurácia alta e ainda assim ser inútil para identificar compras.

## 9. Fazendo uma previsão de cliente novo

```python
novo_cliente = pd.DataFrame([{'idade': 32, 'renda': 5200, 'visitas_site': 6}])
print('Regressão logística:', modelo_logistico.predict(novo_cliente)[0])
print('Probabilidade de compra:', modelo_logistico.predict_proba(novo_cliente)[0, 1])
print('Árvore:', modelo_arvore.predict(novo_cliente)[0])
```

**Explique:** `novo_cliente` precisa ter as **mesmas colunas usadas no treinamento**. É uma simulação de previsão, não uma certeza sobre comportamento humano.

## 10. Auditoria: o modelo funciona de forma semelhante nos grupos?

```python
import numpy as np

auditoria = pd.DataFrame({
    'grupo': grupo_teste.to_numpy(),
    'real': y_teste.to_numpy(),
    'previsto': prev_logistica
})
auditoria['acertou'] = (auditoria['real'] == auditoria['previsto']).astype(int)
print(auditoria.groupby('grupo').agg(
    quantidade=('real', 'size'),
    acuracia=('acertou', 'mean'),
    taxa_prevista_compra=('previsto', 'mean'),
    taxa_real_compra=('real', 'mean')
))

def tpr_do_grupo(parte):
    positivos_reais = parte[parte['real'] == 1]
    if len(positivos_reais) == 0:
        return np.nan
    return (positivos_reais['previsto'] == 1).mean()

print('TPR por grupo:')
print(auditoria.groupby('grupo').apply(tpr_do_grupo, include_groups=False))
```

**Atenção à interpretação:** acurácia por grupo = proporção de acertos em cada grupo; taxa prevista de compra = proporção classificada como 1; taxa real de compra = proporção observada com `comprou=1`; **TPR por grupo** = proporção de compradores reais identificados corretamente em cada grupo. Essas métricas medem coisas diferentes. Uma diferença entre grupos **não prova discriminação**; exige investigação de dados, contexto, finalidade, critérios e consequências.

### Investigando a origem de diferenças

```python
print(df.groupby('grupo').agg(
    clientes=('comprou', 'size'),
    taxa_oferta=('recebeu_oferta', 'mean'),
    taxa_compra=('comprou', 'mean'),
    renda_media=('renda', 'mean')
))
```

Na construção desta base sintética, o grupo A **teve maior chance de receber uma oferta**, e receber uma oferta influenciou a probabilidade simulada de compra. Portanto, diferenças nas compras observadas podem refletir **oportunidades desiguais no histórico**, e não características intrínsecas de A ou B. A variável `grupo` foi excluída do treinamento, mas isso **não garante ausência de viés**: outras variáveis e o próprio alvo histórico podem carregar padrões de desigualdade. Compare taxas e discuta explicações alternativas antes de atribuir causalidade.

## 11. Exercícios e perguntas para a turma

1. O que muda se eu colocar `comprou` dentro de `X`? **Resposta:** vazamento de informação (*target leakage*); o modelo receberia a própria resposta.
2. Qual é a diferença entre `fit()` e `predict()`? **Resposta:** aprender parâmetros/regras versus usar o modelo aprendido para prever.
3. Por que separamos treino e teste? **Resposta:** para estimar desempenho em exemplos não usados no ajuste.
4. Por que a árvore pode acertar tudo no treino e errar no teste? **Resposta:** sobreajuste, especialmente se for profunda demais.
5. Acurácia de 90% significa que o modelo é justo? **Resposta:** não; é necessário verificar erros, desempenho por grupo e consequências.
6. O modelo não recebeu `grupo` como entrada. Isso elimina vieses? **Resposta:** não; outras variáveis podem atuar como proxies, e os resultados históricos podem carregar desigualdades.
7. Por que a taxa de ofertas por grupo é relevante? **Resposta:** mostra que o histórico de oportunidades não foi igual, o que pode afetar o resultado observado.
8. Qual a diferença entre taxa de previsão positiva e TPR? **Resposta:** a primeira considera todos os registros do grupo; TPR considera apenas aqueles cuja classe real é 1.

## 12. Limites e cuidados didáticos

- A base é **fictícia**, com relações geradas artificialmente. Nenhuma diferença observada descreve a realidade de clientes ou grupos sociais.
- `comprou` é um **resultado histórico**, não uma medida universal de mérito, interesse ou capacidade de compra.
- A regressão logística e a árvore podem ter desempenhos diferentes; **não presuma previamente qual vencerá**.
- Uma comparação justa entre algoritmos exige o **mesmo conjunto de teste** e métricas adequadas ao problema.
- **Correlação não é causalidade**; diferenças entre grupos exigem investigação contextual.
- Para uma aplicação real, seriam necessários dados apropriados, avaliação mais robusta, monitoramento e revisão ética.
