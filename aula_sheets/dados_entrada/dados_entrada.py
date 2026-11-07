import os
import json
import gspread
from pathlib import Path
import pandas as pd
from gspread_dataframe import set_with_dataframe
from gspread.exceptions import WorksheetNotFound

# Se a variável de ambiente existir (no GitHub), cria o arquivo temporário
if os.environ.get("GCP_SA_KEY"):
    with open("credentials_temp.json", "w") as f:
        f.write(os.environ["GCP_SA_KEY"])
    cred_file = "credentials_temp.json"
else:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        raise RuntimeError("Configure o secret GCP_SA_KEY com o JSON da conta de serviço no GitHub.")
    # Rodando localmente na sua máquina
    # cred_file = "senai-512823-2179a0ece3af.json"

# --- Leitura dos CSVs ---
PASTA = Path("dados_entrada")
arquivos = sorted(PASTA.glob("*.csv"))

dataframes = []
for arquivo in arquivos:
    df = pd.read_csv(arquivo, dtype={"Pedido": str})
    dataframes.append(df)

df_vendas = pd.concat(dataframes, ignore_index=True)
df_vendas["Total"] = df_vendas["Quantidade"] * df_vendas["Preco_Unitario"]

# --- Conexão ao Google Sheets ---
gc = gspread.service_account(filename=cred_file)
sh = gc.open("HR_Profs_Horistas")

nome_aba = "TESTE"

try:
    worksheet = sh.worksheet(nome_aba)
    worksheet.clear()
except WorksheetNotFound:
    worksheet = sh.add_worksheet(title=nome_aba, rows=len(df_vendas) + 10, cols=len(df_vendas.columns) + 5)

set_with_dataframe(worksheet, df_vendas)

# Limpa o arquivo temporário de credenciais se ele foi criado
if os.path.exists("credentials_temp.json"):
    os.remove("credentials_temp.json")

print("Dados enviados com sucesso!")