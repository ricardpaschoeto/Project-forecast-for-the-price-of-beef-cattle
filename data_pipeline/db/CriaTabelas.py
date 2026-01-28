# Projeto 2 - MLOps da Concepção ao Deploy - Sistema de LLM/RAG
# Python - Pipeline de Criação do Banco de Dados

# Import
import psycopg2

# Função para executar script SQL
def executa_script_sql(filename):
    
    # Conecta ao banco de dados PostgreSQL com as credenciais fornecidas
    conn = psycopg2.connect(
        dbname="metadroiddb",
        user="metadroid",
        password="metadroid1010",
        host="localhost",
        port="5222"
    )

    # Abre um cursor para realizar operações no banco de dados
    cur = conn.cursor()

    # Lê o conteúdo do arquivo SQL
    with open(filename, 'r') as file:
        sql_script = file.read()

    try:
        # Executa o script SQL
        cur.execute(sql_script)

        # Confirma as mudanças no banco de dados
        conn.commit()  
        print("\nScript executado com sucesso!\n")
    except Exception as e:
        # Reverte as mudanças em caso de erro
        conn.rollback()  
        print(f"Erro ao executar o script: {e}")
    finally:
        # Fecha a comunicação com o banco de dados
        cur.close()
        conn.close()

# Executa o script SQL
# executa_script_sql('Projeto2-01-Tabelas.sql')





