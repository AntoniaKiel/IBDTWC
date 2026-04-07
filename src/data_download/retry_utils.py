# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 17:19:12 2025

@author: anton
"""

import time
import logging
from typing import Callable, Any, Optional

def retry_on_connection_error(
    func: Callable, 
    max_retries: Optional[int] = None, 
    retry_delay: float = 180, 
    backoff: float = 1.0, 
    *args, 
    **kwargs
) -> Any:
    """
    Retry a function call if a ConnectionError occurs, with optional delay and backoff.

    Parameters
    ----------
    func : callable
        The function to call. Arguments are passed via *args and **kwargs.
    max_retries : int or None, optional
        Maximum number of retry attempts. Default is None (retry indefinitely).
    retry_delay : float, optional
        Initial delay between retries in seconds. Default is 180.
    backoff : float, optional
        Multiplier for increasing delay after each retry. Default is 1.0 (constant delay).
    *args : positional arguments
        Arguments to pass to `func`.
    **kwargs : keyword arguments
        Keyword arguments to pass to `func`.

    Returns
    -------
    Any
        Return value of `func` if successful.

    Raises
    ------
    ConnectionError
        If the function still raises ConnectionError after the maximum retries.
    """
    attempt = 0
    while True:
        try:
            return func(*args, **kwargs)
        except ConnectionError as e:
            attempt += 1
            logging.info(f"ConnectionError: {e}. Attempt {attempt}{'' if max_retries is None else f'/{max_retries}'}...")
            if max_retries is not None and attempt >= max_retries:
                logging.error("Max retries reached. Aborting.")
                raise
            time.sleep(retry_delay)
            retry_delay *= backoff  # apply backoff for next attempt
            
