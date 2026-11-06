from pathlib import Path
import pandas as pd

# caminho da pasta
PASTA = Path("dados_entrada")
print(PASTA)

# pegando todos os arquivos terminados em .csv
arquivos = sorted(PASTA.glob("*.csv"))
print(arquivos)

# variavel vazia para armazenar o "resultado"
dataframes = []

# para cada arquivo, adiciona na variavel acima
for arquivo in arquivos:
    df = pd.read_csv(arquivo, dtype={"Pedido": str})
    
    dataframes.append(df)

# print(dataframe)

df_vendas = pd.concat(dataframes, ignore_index=True)

print(df_vendas)