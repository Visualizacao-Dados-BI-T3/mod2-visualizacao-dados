# docs: https://github.com/burnash/gspread | https://docs.gspread.org/en/latest/oauth2.html

import gspread

# trocar a chave
gc = gspread.service_account(filename='senai-512823-2179a0ece3af.json')

# trocar nome da planilha
sh = gc.open("Copy of Base de Imóveis")

# sheet1 = "Imóveis"

worksheet = sh.get_worksheet(2)
worksheet.clear()


