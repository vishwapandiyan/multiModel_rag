

import os
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path

def setup_logging(
    log_level: str = "INFO",
    log_file: str = "document_processing/logs/app.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> None:
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert string log level to logging constant
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure root logger
        logging.basicConfig(
            level=numeric_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                # Console handler
                logging.StreamHandler(),
                # File handler with rotation
                logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
            ]
        )
        
        # Set specific logger levels
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        
        # Log successful setup
        logger = logging.getLogger(__name__)
        logger.info(f"Logging configured successfully. Level: {log_level}, File: {log_file}")
        
    except Exception as e:
        # Fallback to basic logging if setup fails
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        logging.error(f"Failed to setup logging: {e}")

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def log_function_call(func_name: str, args: dict = None, kwargs: dict = None):
    """
    Decorator to log function calls
    
    Args:
        func_name: Name of the function being called
        args: Function arguments
        kwargs: Function keyword arguments
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} completed successfully")
                return result
            except Exception as e:
                logger.error(f"{func_name} failed with error: {e}")
                raise
        
        return wrapper
    return decorator

def log_execution_time(func_name: str):
    """
    Decorator to log function execution time
    
    Args:
        func_name: Name of the function being timed
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            start_time = datetime.now()
            
            logger.debug(f"Starting {func_name}")
            
            try:
                result = func(*args, **kwargs)
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                logger.info(f"{func_name} completed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                logger.error(f"{func_name} failed after {execution_time:.2f} seconds with error: {e}")
                raise
        
        return wrapper
    return decorator

def log_memory_usage(func_name: str):
    """
    Decorator to log memory usage before and after function execution
    
    Args:
        func_name: Name of the function being monitored
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            
            try:
                import psutil
                process = psutil.Process()
                
                # Log memory before
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                logger.debug(f"{func_name} - Memory before: {memory_before:.2f} MB")
                
                result = func(*args, **kwargs)
                
                # Log memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_diff = memory_after - memory_before
                
                logger.debug(f"{func_name} - Memory after: {memory_after:.2f} MB (diff: {memory_diff:+.2f} MB)")
                
                return result
            except ImportError:
                # psutil not available, skip memory logging
                logger.debug(f"psutil not available, skipping memory logging for {func_name}")
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error logging memory usage for {func_name}: {e}")
                return func(*args, **kwargs)
        
        return wrapper
    return decorator

def setup_performance_logging(log_file: str = "document_processing/logs/performance.log"):
    """
    Setup separate logging for performance metrics
    
    Args:
        log_file: Path to performance log file
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create performance logger
        perf_logger = logging.getLogger('performance')
        perf_logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not perf_logger.handlers:
            # File handler for performance logs
            perf_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=3,
                encoding='utf-8'
            )
            
            # Performance-specific format
            perf_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            perf_handler.setFormatter(perf_formatter)
            
            perf_logger.addHandler(perf_handler)
        
        logging.getLogger(__name__).info(f"Performance logging configured: {log_file}")
        
    except Exception as e:
        logging.error(f"Failed to setup performance logging: {e}")

def log_performance_metric(metric_name: str, value: float, unit: str = "", context: str = ""):
    """
    Log a performance metric
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        unit: Unit of measurement
        context: Additional context information
    """
    try:
        perf_logger = logging.getLogger('performance')
        
        message = f"PERF_METRIC: {metric_name}={value}"
        if unit:
            message += f" {unit}"
        if context:
            message += f" | Context: {context}"
        
        perf_logger.info(message)
        
    except Exception as e:
        # Fallback to regular logging if performance logging fails
        logging.warning(f"Performance logging failed: {e}")

# Initialize logging when module is imported
setup_logging()
