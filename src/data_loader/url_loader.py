# url_loader.py

import logging
from typing import Any, Optional

import requests
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)

logger = logging.getLogger(__name__)


def load_data_from_url(url: str, timeout: int = 10) -> Optional[Any]:
    """
    URL에서 데이터를 요청하여 반환합니다.

    Args:
        url: 데이터를 불러올 URL
        timeout: 요청 최대 대기 시간(초)

    Returns:
        JSON 응답이면 dict 또는 list를 반환합니다.
        JSON이 아닌 응답이면 문자열을 반환합니다.
        오류가 발생하면 None을 반환합니다.
    """
    try:
        if not isinstance(url, str):
            raise TypeError("URL은 문자열이어야 합니다.")

        url = url.strip()

        if not url:
            raise ValueError("URL은 비어 있을 수 없습니다.")

        if not url.startswith(("http://", "https://")):
            raise ValueError(
                "URL은 http:// 또는 https://로 시작해야 합니다."
            )

        if timeout <= 0:
            raise ValueError("timeout은 0보다 커야 합니다.")

        logger.info("URL 데이터 요청 시작: %s", url)

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            "",
        ).lower()

        if "application/json" in content_type:
            try:
                data = response.json()
                logger.info("JSON 데이터 로드 성공: %s", url)
                return data

            except requests.exceptions.JSONDecodeError as error:
                logger.error(
                    "JSON 데이터 변환 실패: %s, 오류: %s",
                    url,
                    error,
                )
                return None

        logger.info("텍스트 데이터 로드 성공: %s", url)
        return response.text

    except TypeError as error:
        logger.error("URL 타입 오류: %s", error)
        return None

    except ValueError as error:
        logger.error("입력값 오류: %s", error)
        return None

    except Timeout:
        logger.error(
            "요청 시간 초과: %s, timeout=%s초",
            url,
            timeout,
        )
        return None

    except ConnectionError:
        logger.error("서버 연결 실패: %s", url)
        return None

    except HTTPError as error:
        status_code = (
            error.response.status_code
            if error.response is not None
            else "알 수 없음"
        )

        logger.error(
            "HTTP 오류 발생: URL=%s, 상태 코드=%s",
            url,
            status_code,
        )
        return None

    except RequestException as error:
        logger.error(
            "URL 요청 중 오류 발생: %s, 오류: %s",
            url,
            error,
        )
        return None

    except Exception as error:
        logger.exception(
            "예상하지 못한 오류 발생: %s",
            error,
        )
        return None

    finally:
        logger.info("URL 데이터 요청 종료")
        
