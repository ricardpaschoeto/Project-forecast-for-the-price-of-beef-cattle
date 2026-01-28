# Projeto 2 - MLOps da Concepção ao Deploy - Sistema de LLM/RAG
# Python e SQL - Pipeline de Carga de Dados

# Imports
import psycopg2
import pandas as pd
from sqlalchemy import create_engine

# Cria o motor de conexão
engine = create_engine('postgresql+psycopg2://dsa:dsa1010@localhost:5222/metadroiddb')

print("\nIniciando o Processo de Carga dos Dados!\n")

# Função para carregar dados dos arquivos CSV para o PostgreSQL no schema especificado
def carrega_dados(csv_file, table_name, schema):

    # Bloco try/except
    try:

        # Lê o arquivo CSV
        df = pd.read_csv(csv_file)

        # Executa SQL a partir do dataframe do Pandas
        df.to_sql(table_name, engine, schema = schema, if_exists = 'append', index = False)
        print(f"Dados do arquivo {csv_file} foram inseridos na tabela {schema}.{table_name}.")

    except Exception as e:
        print(f"Erro ao inserir dados do arquivo {csv_file} na tabela {schema}.{table_name}: {e}")

# Carregamento dos dados no schema 'projeto5'
#dsa_carrega_dados('dsa_clientes.csv', 'clientes', 'projeto2')
#dsa_carrega_dados('dsa_produtos.csv', 'produtos_eletronicos', 'projeto2')
#dsa_carrega_dados('dsa_vendas.csv', 'vendas', 'projeto2')

#print("\nCarga Executada com Sucesso! Use o pgAdmin Para Checar os Dados Se Desejar!\n")
#print("\nIniciando o Processo de Análise dos Dados com IA. Seja Paciente e Aguarde o Excelente Resultado Que Será Entregue a Você!\n")















