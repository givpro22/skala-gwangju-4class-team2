import logging
from io import StringIO
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import polars as pl

from src.data_loader.url_loader import load_data_from_url


logger = logging.getLogger(__name__)


def load_dataframes_from_url(
    url: str,
    timeout: int = 10,
    pandas_options: Optional[Dict[str, Any]] = None,
    polars_options: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[pd.DataFrame], Optional[pl.DataFrame]]:
    """URL의 CSV 데이터를 Pandas와 Polars DataFrame으로 읽습니다.

    URL 요청은 한 번만 수행합니다. 라이브러리별 ``read_csv`` 옵션은
    ``pandas_options``와 ``polars_options``에 각각 전달할 수 있습니다.
    로딩이나 파싱에 실패하면 두 결과 모두 ``None``을 반환합니다.
    """
    data = load_data_from_url(url, timeout=timeout)

    if data is None:
        return None, None

    if not isinstance(data, str):
        logger.error("CSV 텍스트가 아닌 응답을 받았습니다: %s", url)
        return None, None

    try:
        pandas_df = pd.read_csv(
            StringIO(data),
            **(pandas_options or {}),
        )
        polars_df = pl.read_csv(
            StringIO(data),
            **(polars_options or {}),
        )
    except (
        OSError,
        TypeError,
        ValueError,
        pd.errors.ParserError,
        pl.exceptions.PolarsError,
    ) as error:
        logger.error("DataFrame 변환 실패: %s", error)
        return None, None

    logger.info(
        "DataFrame 로드 성공: pandas=%s, polars=%s",
        pandas_df.shape,
        polars_df.shape,
    )
    return pandas_df, polars_df
