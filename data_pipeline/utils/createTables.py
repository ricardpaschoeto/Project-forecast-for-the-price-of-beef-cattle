# Import
import psycopg2

# Função para executar script SQL

def executa_script_sql(filename):
    conn = None  # Inicializa como None
    try:
        conn = psycopg2.connect(
            database="metadroiddb",
            user="postgres",
            password="metadroid1010",
            host="localhost",
            port="5432"
        )
        cur = conn.cursor()
        with open(filename, 'r') as file:
            sql_script = file.read()
        cur.execute(sql_script)
        conn.commit()
        print("\nScript executado com sucesso!\n")
    except Exception as e:
        if conn:  # Só faz rollback se a conexão foi criada
            conn.rollback()
        print("Erro ao executar o script:", e)
    finally:
        if conn:
            cur.close()
            conn.close()

# Executa o script SQL
executa_script_sql('D:/Users/Ricar/OneDrive/Documentos/PROJETOS_ML_DATA_SCIENCE/PROJETO_AGRO_GOIAS/PROJETO_01/data_pipeline/utils/Table.sql')





