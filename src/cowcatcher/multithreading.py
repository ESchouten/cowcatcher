import threading
import time
import logging
from abc import ABC, abstractmethod


class MultiThreading(ABC):
    """
    Base class for tasks that run periodically in a separate thread.
    """
    
    def __init__(self, interval: float = 2.0):
        """
        Initialize periodic task.
        
        Args:
            interval: Time between task executions in seconds
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.interval = interval
        
        self._running = False
        self._thread = None
        self._task_lock = threading.Lock()
        
        
        # Statistics
        self._execution_count = 0
        self._skipped_count = 0
    
    def start_run(self):
        """Start the periodic task in a separate thread."""
        if self._running:
            self.logger.warning("Task is already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._task_loop, daemon=True)
        self._thread.start()
        self.logger.info("Periodic task thread started")
    
    def stop_run(self):
        """Stop the periodic task thread."""
        if not self._running:
            self.logger.warning("Task is not running")
            return
        
        self.logger.info("Stopping task thread...")
        self._running = False
        
        if self._thread:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                self.logger.warning("Thread did not stop gracefully")
            else:
                self.logger.info("Task thread stopped successfully")
        
        self._thread = None
        self._log_statistics()
    
    def _task_loop(self):
        """Internal task loop (runs in separate thread)."""
        self.logger.info("Task loop started")
        
        try:
            while self._running:
                # Try to acquire lock
                if self._task_lock.acquire(blocking=False):
                    try:
                        start_time = time.time()
                        self.execute()  # Call the method in the child class to execute AI detection
                        duration = time.time() - start_time
                        
                        self._execution_count += 1
                        self.logger.debug(
                            f"Execution #{self._execution_count} completed in {duration:.2f}s"
                        )
                        
                        # Warning if execution takes too long
                        if duration > (self.interval * 0.9):
                            self.logger.warning(
                                f"Execution took {duration:.2f}s - close to interval of {self.interval}s!"
                            )
                    finally:
                        self._task_lock.release()
                else:
                    # Previous execution still running
                    self._skipped_count += 1
                    self.logger.warning(
                        f"Previous execution still running, skipping (total skipped: {self._skipped_count})"
                    )
                
                # Wait for interval (check every 0.1s if we need to stop)
                self._wait_interval()
                
        except Exception as e:
            self.logger.exception(f"Error in task loop: {e}")
        finally:
            self.logger.info("Task loop stopped")
            # self._log_statistics()
    
    def _wait_interval(self):
        """Wait for the interval while checking if we need to stop."""
        steps = int(self.interval / 0.1)
        for _ in range(steps):
            if not self._running:
                break
            time.sleep(0.1)
    
    def _log_statistics(self):
        """Log execution statistics."""
        # self.logger.info(
        #     f"Statistics: {self._execution_count} executions, {self._skipped_count} skipped"
        stats = self.get_stats()
        self.logger.info(f"Statistics: {stats}")
        
    
    def is_running(self) -> bool:
        """Check if task is running."""
        return self._running
    
    def get_stats(self) -> dict:
        """Get task statistics."""
        return {
            'execution_count': self._execution_count,
            'skipped_count': self._skipped_count,
            'is_running': self._running,
            'interval': self.interval
        }
    
    @abstractmethod
    def execute(self):
        """
        Execute the task. Must be implemented by subclass.
        This method is called periodically in a separate thread.
        """
        pass