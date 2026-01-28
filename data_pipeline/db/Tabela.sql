-- Projeto 2 - MLOps da Concepção ao Deploy - Sistema de LLM/RAG
-- SQL - Criação do Banco de Dados

-- Deleta o schema se já existir
-- DROP SCHEMA IF EXISTS boi_gordo CASCADE;

-- Cria o schema
CREATE SCHEMA IF NOT EXISTS boi_gordo;

-- Cria as tabelas

CREATE TABLE boi_gordo.data (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    idade INTEGER NOT NULL,
    email VARCHAR(100) NOT NULL UNIQUE,
    telefone VARCHAR(20) NOT NULL,
    cidade VARCHAR(100) NOT NULL
);
