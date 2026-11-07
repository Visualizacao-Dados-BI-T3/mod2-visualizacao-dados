from pathlib import Path
import pandas as pd
import gspread
from gspread_dataframe import set_with_dataframe

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

# print(df_vendas)

# print(df_vendas.info())

# df["Quantidade"] = pd.to_numeric(df["Quantidade"], errors="coerce")

print(df_vendas.info())

df_vendas["Total"] = df_vendas["Quantidade"] * df_vendas["Preco_Unitario"]

print(df_vendas)

# -----------------------------

gc = gspread.service_account(filename='senai-512823-2179a0ece3af.json')

sh = gc.open("HR_Profs_Horistas")

# worksheet = sh.get_worksheet(0)
# print(worksheet.get_values())

# worksheet = sh.add_worksheet(title="b", rows=100, cols=20)

worksheet = sh.get_worksheet(2)

set_with_dataframe(worksheet, df_vendas)