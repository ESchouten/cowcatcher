from abc import ABC, abstractmethod

class NotifierClass(ABC):
    """Base class voor alle notifiers"""
    
    @abstractmethod
    def _test_connection(self) -> bool:
        """Test of de verbinding met de notificatie service werkt"""
        pass
    
    @abstractmethod
    def send_photo(self, image_path: str, caption: str, disable_notification: bool = False) -> bool:
        """Verstuur een foto met caption"""
        pass
    
    @abstractmethod
    def send_message(self, message: str) -> bool:
        """Verstuur een tekstbericht"""
        pass