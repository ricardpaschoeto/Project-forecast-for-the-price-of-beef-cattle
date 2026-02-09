"""clean_pipeline.py

Pipeline de limpeza de dados e engenharia de atributos.

Este módulo foi refatorado para organizar as funções originais em uma classe,
com responsabilidades bem definidas, tipagem, documentação e logs.

Comportamento preservado (na essência):
- Linhas com NA em colunas inteiras são removidas (dropna).
- Em colunas float, valores faltantes são preenchidos por *forward fill* (ffill)
  e, se ainda restarem NAs nessas colunas, as linhas correspondentes são removidas.
- Colunas categóricas podem ser codificadas via one-hot (get_dummies) ou
  codificação ordinal (mapeamento).

Uso rápido:
    python clean_pipeline.py

Uso programático:
    from clean_pipeline import DataCleanPipeline
    pipeline = DataCleanPipeline(input_path='dados_agro_atualizado.csv')
    df = pipeline.run()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt='[%(levelname)s] %(name)s - %(message)s'
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Configuração
# -----------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """Configurações do pipeline.

    Attributes
    ----------
    input_path:
        Caminho do arquivo CSV de entrada.
    output_path:
        Caminho do arquivo CSV de saída.
    categorical_onehot:
        Lista de colunas categóricas a serem codificadas com one-hot.
    categorical_ordinal:
        Coluna categórica a ser codificada ordinalmente.
    ordinal_mapping:
        Dicionário de mapeamento para codificação ordinal.
    save_index:
        Se True, salva o índice no CSV de saída.
    """

    input_path: str
    output_path: str
    categorical_onehot: List[str] = field(default_factory=lambda: ['ciclo', 'aspecto'])
    categorical_ordinal: str = 'el nino'
    ordinal_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            'INEXISTENTE': 0,
            'FRACO': 1,
            'MÉDIO': 2,
            'FORTE': 3,
        }
    )
    save_index: bool = True


class DataCleanPipeline:
    """Pipeline de limpeza e codificação de variáveis.

    Esta classe encapsula as funções originais do arquivo em uma API orientada a
    objetos, permitindo reuso, testes e parametrização.

    Parameters
    ----------
    input_path:
        Caminho para o CSV de entrada.
    output_path:
        Caminho para o CSV de saída.
    config:
        Configuração opcional. Se fornecida, sobrescreve input/output passados.
    log_level:
        Nível de log do módulo (ex.: logging.INFO).

    Notes
    -----
    - O pipeline detecta automaticamente colunas int64 e float64.
    - A limpeza de int64 é feita por remoção de linhas com NA nessas colunas.
    - O preenchimento de float64 é feito por forward-fill (ffill).
    """

    def __init__(
        self,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        config: Optional[PipelineConfig] = None,
        log_level: int = logging.INFO,
    ) -> None:
        logger.setLevel(log_level)

        if config is None:
            if input_path is None or output_path is None:
                raise ValueError(
                    'Informe input_path e output_path ou forneça um PipelineConfig.'
                )
            self.config = PipelineConfig(input_path=input_path, output_path=output_path)
        else:
            # Se config for passado, ele manda.
            self.config = config

    # ---------------------------------------------------------------------
    # IO
    # ---------------------------------------------------------------------
    def read(self) -> pd.DataFrame:
        """Lê o CSV de entrada."""
        logger.info('Lendo dados: %s', self.config.input_path)
        return pd.read_csv(self.config.input_path)

    def save(self, df: pd.DataFrame) -> None:
        """Salva o DataFrame no CSV de saída."""
        logger.info('Salvando dados: %s', self.config.output_path)
        df.to_csv(self.config.output_path, index=self.config.save_index)

    # ---------------------------------------------------------------------
    # Etapas do pipeline (equivalentes às funções originais)
    # ---------------------------------------------------------------------
    @staticmethod
    def drop_missing_rows(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Remove linhas com valores ausentes nas colunas informadas.

        Esta etapa corresponde ao comportamento original de `_clean_data`.

        Parameters
        ----------
        df:
            DataFrame de entrada.
        columns:
            Colunas a serem verificadas.

        Returns
        -------
        pd.DataFrame
            DataFrame sem linhas com NA nas colunas alvo.
        """
        if not columns:
            return df

        before = len(df)
        df = df.dropna(subset=list(columns))
        after = len(df)
        logger.info('drop_missing_rows: removidas %s linhas (de %s para %s).', before - after, before, after)

        # Sanity check
        na_count = int(df[list(columns)].isna().sum().sum())
        if na_count == 0:
            logger.info('drop_missing_rows: nenhuma NA restante nas colunas alvo.')
        else:
            logger.warning('drop_missing_rows: ainda existem %s NAs nas colunas alvo.', na_count)
        return df

    @staticmethod
    def forward_fill(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
        """Preenche valores ausentes com *forward fill* nas colunas informadas.

        Esta etapa corresponde ao comportamento original de `_fill_data`.
        Após o ffill, remove linhas que ainda tenham NA nessas colunas.

        Parameters
        ----------
        df:
            DataFrame de entrada.
        columns:
            Colunas nas quais aplicar o preenchimento.

        Returns
        -------
        pd.DataFrame
            DataFrame com NAs tratados nas colunas alvo.
        """
        if not columns:
            return df

        cols = list(columns)
        for col in cols:
            df[col] = df[col].ffill()

        before = len(df)
        df = df.dropna(subset=cols)
        after = len(df)
        logger.info('forward_fill: removidas %s linhas após ffill (de %s para %s).', before - after, before, after)

        na_count = int(df[cols].isna().sum().sum())
        if na_count == 0:
            logger.info('forward_fill: nenhuma NA restante nas colunas alvo.')
        else:
            logger.warning('forward_fill: ainda existem %s NAs nas colunas alvo.', na_count)
        return df

    @staticmethod
    def encode_onehot(
        df: pd.DataFrame,
        columns: Sequence[str],
        drop_first: bool = True,
        dtype: type = int,
    ) -> pd.DataFrame:
        """Codifica colunas categóricas usando one-hot encoding."""
        if not columns:
            return df

        try:
            return pd.get_dummies(df, columns=list(columns), drop_first=drop_first, dtype=dtype)
        except Exception as exc:
            logger.exception('Erro no one-hot encoding: %s', exc)
            raise

    def encode_ordinal(
        self,
        df: pd.DataFrame,
        column: str,
        mapping: Optional[Dict[str, int]] = None,
        suffix: str = '_encoded',
        drop_original: bool = True,
    ) -> pd.DataFrame:
        """Codifica uma coluna categórica ordinalmente.

        Cria uma nova coluna `<column><suffix>` com os valores mapeados.

        Parameters
        ----------
        df:
            DataFrame de entrada.
        column:
            Nome da coluna a codificar.
        mapping:
            Dicionário opcional de mapeamento. Se None, usa o do config.
        suffix:
            Sufixo para o nome da nova coluna.
        drop_original:
            Se True, remove a coluna original.
        """
        if not column:
            return df

        mapping = mapping or self.config.ordinal_mapping
        new_col = f'{column}{suffix}'

        try:
            df[new_col] = df[column].map(mapping)
            if drop_original:
                df = df.drop(columns=[column])
        except Exception as exc:
            logger.exception('Erro na codificação ordinal (%s): %s', column, exc)
            raise

        return df

    # ---------------------------------------------------------------------
    # Orquestração
    # ---------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        """Executa o pipeline completo e retorna o DataFrame final."""
        df = self.read()

        int_cols = df.select_dtypes(include=['int64']).columns.tolist()
        float_cols = df.select_dtypes(include=['float64']).columns.tolist()

        logger.info('Colunas int64 detectadas: %s', int_cols)
        logger.info('Colunas float64 detectadas: %s', float_cols)

        df = self.drop_missing_rows(df, int_cols)
        df = self.forward_fill(df, float_cols)

        df = self.encode_onehot(df, self.config.categorical_onehot)
        df = self.encode_ordinal(df, self.config.categorical_ordinal)

        self.save(df)
        return df


def _default_paths() -> tuple[str, str]:
    """Mantém a convenção de caminhos relativos ao diretório do script."""
    actual_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(actual_dir, 'dados_agro_atualizado.csv')
    output_path = os.path.join(actual_dir, 'dados_limpos.csv')
    return input_path, output_path


def clean_pipeline() -> pd.DataFrame:
    """Função de compatibilidade com a versão anterior."""
    input_path, output_path = _default_paths()
    pipeline = DataCleanPipeline(input_path=input_path, output_path=output_path)
    return pipeline.run()


if __name__ == '__main__':
    clean_pipeline()
